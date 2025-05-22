#!/usr/bin/env python3
"""
criu_diff.py – byte-accurate diff between two CRIU dumps
Assumptions:
  • both dumps were taken with --track-mem (incremental)
  • neither dump nor its ancestors were passed through --auto-dedup
  • page size is the kernel base page (usually 4096)
"""

import argparse, json, mmap, os, subprocess, sys
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")  # 4096 on x86/amd64


def decode_pagemap(img_path: Path) -> list[dict]:
    """
    Run "crit decode" and return a flat list of dicts
    having keys addr, pages, in_parent.
    """
    raw = subprocess.check_output(
        ["python", "-m", "crit", "decode", "-i", str(img_path)],
        text=True,
    )
    root = json.loads(raw)
    entries = []
    for outer in root["entries"]:
        inner = next(iter(outer.values()))  # strip {"pagemap": {...}}
        entries.append(inner)
    return entries


def build_index(dump_dir: Path) -> tuple[dict[int, int], mmap.mmap]:
    """
    Returns:
        idx – dict {vaddr: file_offset}
        buf – mmap of pages-1.img
    Only pages with in_parent == False are indexed (they exist in this layer).
    """
    pmap = decode_pagemap(dump_dir / "pagemap-1.img")
    f = open(dump_dir / "pages-1.img", "rb")
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    offset = 0
    idx = {}
    for ent in pmap:
        for i in range(ent["pages"]):
            vaddr = ent["addr"] + i * PAGE
            if not ent.get("in_parent", False):
                idx[vaddr] = offset
            offset += PAGE
    return idx, buf


def popcount_diff(a: memoryview, b: memoryview) -> int:
    """Return the number of differing bytes between two 4 KiB pages."""
    diff_bits = 0
    for off in range(0, PAGE, 8):  # 8-byte blocks
        chunk_a = int.from_bytes(a[off : off + 8], "little", signed=False)
        chunk_b = int.from_bytes(b[off : off + 8], "little", signed=False)
        diff_bits += (chunk_a ^ chunk_b).bit_count()
    return diff_bits // 8  # 8 bits per byte


def main():
    ap = argparse.ArgumentParser(description="Byte diff between CRIU dumps")
    ap.add_argument("parent", type=Path, help="earlier checkpoint dir")
    ap.add_argument("child", type=Path, help="later   checkpoint dir")
    args = ap.parse_args()

    parent_idx, parent_buf = build_index(args.parent)
    child_idx, child_buf = build_index(args.child)

    zero_page = bytes(PAGE)
    bytes_diff = 0
    pages_touched = 0

    all_addrs = parent_idx.keys() | child_idx.keys()
    for addr in all_addrs:
        if addr in child_idx:
            cpage = memoryview(child_buf)[child_idx[addr] : child_idx[addr] + PAGE]
        else:
            cpage = zero_page

        if addr in parent_idx:
            ppage = memoryview(parent_buf)[parent_idx[addr] : parent_idx[addr] + PAGE]
        else:
            ppage = zero_page

        if cpage is ppage:  # same object when both are zero_page
            continue

        delta = popcount_diff(cpage, ppage)
        if delta:
            bytes_diff += delta
            pages_touched += 1

    print(f"bytes_diff={bytes_diff}  pages_touched={pages_touched}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"crit decode failed: {e}")
