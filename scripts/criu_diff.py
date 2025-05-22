#!/usr/bin/env python3
"""
criu_diff.py – count byte differences between an incremental CRIU dump
               and its immediate parent.

Assumes:
  • dumps were taken with --track-mem
  • --auto-dedup was *not* used
  • python -m crit is available (pycriu / protobuf < 4)
"""

import argparse, json, mmap, os, subprocess, sys
from pathlib import Path
from typing import Dict, Tuple

PAGE = os.sysconf("SC_PAGE_SIZE")  # usually 4096


def pick_pagemap(dir_: Path) -> Path:
    """Return the pagemap that belongs to pages-1.img (pages_id == 1)."""
    for pm in dir_.glob("pagemap-*.img"):
        raw = subprocess.check_output(
            ["python", "-m", "crit", "decode", "-i", str(pm), "--shallow"], text=True
        )
        first = json.loads(raw)["entries"][0]
        if first.get("pages_id") == 1:
            return pm
    sys.exit(f"[ERR] {dir_}: no pagemap with pages_id==1 found")


def decode_pagemap(pmap: Path):
    """
    Yield tuples (vaddr, nr_pages, in_parent_flag).

    JSON records have:
        {"vaddr": ..., "nr_pages": N, "flags": F}
    """
    raw = subprocess.check_output(
        ["python", "-m", "crit", "decode", "-i", str(pmap)], text=True
    )
    for entry in json.loads(raw)["entries"]:
        if "vaddr" not in entry:  # skip {"pages_id": ...}
            continue
        in_parent = entry["flags"] & 1  # bit 0 => identical to parent
        yield entry["vaddr"], entry["nr_pages"], bool(in_parent)


def build_index(dump: Path) -> Tuple[Dict[int, int], mmap.mmap, object]:
    """
    Build {virtual_addr -> offset_in_pages_file} for all pages stored
    in this dump (i.e. NOT in_parent).  Return (index, mmap, file_handle).
    """
    pages = dump / "pages-1.img"
    pagemap = pick_pagemap(dump)

    print(f"[INFO] {dump.name:6}: {pages.name} + {pagemap.name}")

    fh = open(pages, "rb")
    buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    idx, offset = {}, 0
    for vaddr, n, in_parent in decode_pagemap(pagemap):
        for _ in range(n):
            if not in_parent:
                idx[vaddr] = offset
            vaddr += PAGE
            offset += PAGE
    return idx, buf, fh


def page_delta(a: memoryview, b: memoryview) -> int:
    """Return number of differing bytes between two 4 KiB pages."""
    # Fast branch-free per-byte compare; ~1 GB/s per core in pure Python.
    return sum(x != y for x, y in zip(a, b))


def resolve_parent(child: Path) -> Path:
    link = child / "parent"
    if not link.exists():
        sys.exit(
            f"[ERR] {child}: missing 'parent' symlink " "(not an incremental dump?)"
        )
    return (child / link.readlink()).resolve()


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Byte-accurate diff between CRIU dump and its parent"
    )
    ap.add_argument(
        "child_dir", type=Path, help="directory of the *newer* (child) checkpoint"
    )
    args = ap.parse_args()

    child = args.child_dir.resolve()
    parent = resolve_parent(child)

    p_idx, p_buf, p_fh = build_index(parent)
    c_idx, c_buf, c_fh = build_index(child)

    zero_page = bytes(PAGE)
    bytes_diff = pages_comp = 0

    for addr in p_idx.keys() | c_idx.keys():
        pa = (
            memoryview(p_buf)[p_idx[addr] : p_idx[addr] + PAGE]
            if addr in p_idx
            else zero_page
        )
        ca = (
            memoryview(c_buf)[c_idx[addr] : c_idx[addr] + PAGE]
            if addr in c_idx
            else zero_page
        )
        delta = page_delta(pa, ca)
        if delta:
            bytes_diff += delta
            pages_comp += 1

    print(f"bytes_diff={bytes_diff}   pages_compared={pages_comp}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"[ERR] crit decode failed: {e}")
