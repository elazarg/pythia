#!/usr/bin/env python3
"""
criu_diff.py – byte-accurate RAM delta between two CRIU dump dirs
  • works with --track-mem    (incremental pages-1.img + pagemap-*.img)
  • assumes --auto-dedup was NOT used
  • tolerant of CRIU's per-mm suffixes like pagemap-167470.img
"""

import argparse, json, mmap, os, subprocess, sys
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")  # 4096 on x86/amd64


# ---------- helpers ---------------------------------------------------------


def choose_single(dir_: Path, glob_pat: str, what: str) -> Path:
    """Return exactly one file matching the pattern or abort."""
    matches = sorted(dir_.glob(glob_pat))
    if not matches:
        sys.exit(f"[ERR] {dir_}: no {what} ({glob_pat}) found")
    if len(matches) > 1:
        # Heuristic: the root-process pagemap is almost always the biggest.
        matches.sort(key=lambda p: p.stat().st_size, reverse=True)
    return matches[0]


def decode_pagemap(pagemap: Path):
    """Run `crit decode` and yield (addr, pages, in_parent) dicts."""
    out = subprocess.check_output(
        ["python", "-m", "crit", "decode", "-i", str(pagemap)],
        text=True,
    )
    root = json.loads(out)
    for outer in root["entries"]:
        yield next(iter(outer.values()))  # strip outer key


def build_index(dump_dir: Path) -> tuple[dict[int, int], mmap.mmap]:
    """
    Return (index, buf) where:
        index : {virtual_addr -> offset_in_pages_file}
        buf   : mmap of pages file (read-only)
    """
    pages = choose_single(dump_dir, "pages-*.img", "pages image")
    pagemap = choose_single(dump_dir, "pagemap-*.img", "pagemap image")

    print(f"[INFO] {dump_dir.name}: using {pages.name}  +  {pagemap.name}")

    buf = mmap.mmap(open(pages, "rb").fileno(), 0, access=mmap.ACCESS_READ)

    idx, off = {}, 0
    for e in decode_pagemap(pagemap):
        for i in range(e["pages"]):
            if not e.get("in_parent", False):
                idx[e["addr"] + i * PAGE] = off
            off += PAGE
    return idx, buf


def popcnt64(x: int) -> int:  # Python 3.8 + has int.bit_count()
    return x.bit_count()


def bytes_diff(a: memoryview, b: memoryview) -> int:
    """Return number of differing bytes between two 4 KiB pages."""
    diff_bits = 0
    for o in range(0, PAGE, 8):
        diff_bits += popcnt64(
            int.from_bytes(a[o : o + 8], "little")
            ^ int.from_bytes(b[o : o + 8], "little")
        )
    return diff_bits >> 3  # /8


def main():
    argp = argparse.ArgumentParser(description="Diff two CRIU RAM dumps")
    argp.add_argument("parent", type=Path)
    argp.add_argument("child", type=Path)
    ns = argp.parse_args()

    p_idx, p_buf = build_index(ns.parent.resolve())
    c_idx, c_buf = build_index(ns.child.resolve())

    zero_page = bytes(PAGE)
    delta_bytes = 0
    pages_compared = 0

    for addr in p_idx.keys() | c_idx.keys():
        ap = (
            memoryview(p_buf)[p_idx[addr] : p_idx[addr] + PAGE]
            if addr in p_idx
            else zero_page
        )
        ac = (
            memoryview(c_buf)[c_idx[addr] : c_idx[addr] + PAGE]
            if addr in c_idx
            else zero_page
        )
        if ap is ac:  # same object when both zero_page
            continue
        delta = bytes_diff(ap, ac)
        if delta:
            delta_bytes += delta
            pages_compared += 1

    print(f"bytes_diff={delta_bytes}  pages_compared={pages_compared}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"[ERR] crit decode failed: {e}")
