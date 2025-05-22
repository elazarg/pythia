#!/usr/bin/env python3
"""
criu_diff.py  –  byte-accurate RAM delta for one CRIU checkpoint

  * pass exactly ONE dump directory (the “child”)
  * the script follows the  ‘parent’  symlink created by CRIU
  * dumps were taken with  --track-mem   and  NO  --auto-dedup
  * requires  python -m crit   in $PATH   (pycriu / protobuf < 4)

Outputs:
    bytes_diff=<N>   pages_compared=<M>
"""

import argparse, json, mmap, os, subprocess, sys
from pathlib import Path
from typing import Dict, Tuple

PAGE = os.sysconf("SC_PAGE_SIZE")  # 4096 on x86_64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def pick_one(dir_: Path, pattern: str, role: str) -> Path:
    """Return exactly one file matching pattern inside dir_ (largest if many)."""
    matches = sorted(dir_.glob(pattern))
    if not matches:
        sys.exit(f"[ERR] {dir_}: no {role} ({pattern}) found")
    if len(matches) > 1:
        matches.sort(key=lambda p: p.stat().st_size, reverse=True)
    return matches[0]


def decode_pagemap(pmap: Path):
    """
    Yield dictionaries with keys  addr, pages, (optional) in_parent
    """
    raw = subprocess.check_output(
        ["python", "-m", "crit", "decode", "-i", str(pmap)],
        text=True,
    )
    root = json.loads(raw)
    # In current CRIU JSON each entry is already the dict we want
    for entry in root["entries"]:
        yield entry  # nothing to unwrap


def build_index(dump_dir: Path) -> Tuple[Dict[int, int], mmap.mmap, object]:
    """
    Build {vaddr -> file_offset} index for pages stored in this dump.
    Returns (index, mmap_object, file_handle_to_keep_alive).
    """
    pages = pick_one(dump_dir, "pages-*.img", "pages image")
    pagemap = pick_one(dump_dir, "pagemap-*.img", "pagemap image")

    print(f"[INFO] {dump_dir.name:7}: {pages.name} + {pagemap.name}")

    fh = open(pages, "rb")
    buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    idx, off = {}, 0
    for e in decode_pagemap(pagemap):
        for i in range(e["pages"]):
            if not e.get("in_parent", False):
                idx[e["addr"] + i * PAGE] = off
            off += PAGE
    return idx, buf, fh


def page_diff(a: memoryview, b: memoryview) -> int:
    """Return number of differing bytes between two 4 KiB pages."""
    # Fast path: identical object (both zero_page) → 0
    if a is b:
        return 0
    # Compare per-byte; numpy/popcnt could be dropped in if needed.
    return sum(a[i] != b[i] for i in range(PAGE))


def resolve_parent(child_dir: Path) -> Path:
    link = child_dir / "parent"
    if not link.exists():
        sys.exit(
            f"[ERR] {child_dir}: missing 'parent' symlink – "
            "take an incremental dump or specify a parent manually."
        )
    return (child_dir / link.readlink()).resolve()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Diff CRIU incremental dumps")
    parser.add_argument("child_dir", type=Path, help="latest dump directory")
    args = parser.parse_args()

    child = args.child_dir.resolve()
    parent = resolve_parent(child)

    p_idx, p_buf, p_fh = build_index(parent)
    c_idx, c_buf, c_fh = build_index(child)

    zero_pg = bytes(PAGE)
    bytes_diff = pages_compared = 0

    for vaddr in p_idx.keys() | c_idx.keys():
        pa = (
            memoryview(p_buf)[p_idx[vaddr] : p_idx[vaddr] + PAGE]
            if vaddr in p_idx
            else zero_pg
        )
        ca = (
            memoryview(c_buf)[c_idx[vaddr] : c_idx[vaddr] + PAGE]
            if vaddr in c_idx
            else zero_pg
        )
        delta = page_diff(pa, ca)
        if delta:
            bytes_diff += delta
            pages_compared += 1

    print(f"bytes_diff={bytes_diff}   pages_compared={pages_compared}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as err:
        sys.exit(f"[ERR] crit decode failed: {err}")
