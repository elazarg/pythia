#!/usr/bin/env python3
"""
criu_diff.py  –  byte-accurate RAM delta between two CRIU dumps.

Assumptions
-----------
• Dumps were created with --track-mem       (incremental).
• --auto-dedup was NOT used, so the immediate parent still owns all pages.
• Only the primary process address space is of interest (largest pagemap).

Run:
    python criu_diff.py CHILD_DIR             # parent taken from CHILD_DIR/parent
    python criu_diff.py PARENT_DIR CHILD_DIR  # explicit pair
"""

import argparse, json, mmap, os, subprocess, sys
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")  # 4096 on x86/amd64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def pick_one(dir_: Path, pattern: str, what: str) -> Path:
    """Pick exactly one file matching pattern inside dir_ (largest if several)."""
    matches = sorted(dir_.glob(pattern))
    if not matches:
        sys.exit(f"[ERR] {dir_}: no {what} ({pattern}) found")
    if len(matches) > 1:
        matches.sort(key=lambda p: p.stat().st_size, reverse=True)
    return matches[0]


def decode_pagemap(path: Path):
    """Iterate over (addr, pages, in_parent) entries from pagemap-*.img."""
    out = subprocess.check_output(
        ["python", "-m", "crit", "decode", "-i", str(path)],
        text=True,
    )
    root = json.loads(out)
    for wrapper in root["entries"]:
        yield next(iter(wrapper.values()))  # strip outer key ("pagemap")


def build_index(dump_dir: Path):
    """
    Return (index, buf, file_obj) where
        index : {vaddr → offset}
        buf   : mmap object of pages file (read-only)
        file_obj keeps the FD alive for the mmap lifetime.
    """
    pages = pick_one(dump_dir, "pages-*.img", "pages image")
    pagemap = pick_one(dump_dir, "pagemap-*.img", "pagemap image")

    print(f"[INFO] {dump_dir.name:7}: {pages.name}  +  {pagemap.name}")

    f = open(pages, "rb")
    buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    idx, off = {}, 0
    for e in decode_pagemap(pagemap):
        for i in range(e["pages"]):
            if not e.get("in_parent", False):
                idx[e["addr"] + i * PAGE] = off
            off += PAGE
    return idx, buf, f


def page_diff(a: memoryview, b: memoryview) -> int:
    """Return number of differing bytes between two 4 KiB pages."""
    diff_bits = 0
    for o in range(0, PAGE, 8):  # 64-bit chunks
        diff_bits += int.from_bytes(a[o : o + 8], "little") ^ int.from_bytes(
            b[o : o + 8], "little"
        )
    # popcount via Python 3.8+ int.bit_count().  Each set bit ⇒ 1 byte diff?
    # We need per-byte, not per-bit.  A simple, branch-free way:
    return sum(a[i] != b[i] for i in range(PAGE))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def resolve_parent(child_dir: Path) -> Path:
    link = child_dir / "parent"
    if not link.exists():
        sys.exit(f"[ERR] {child_dir}: no 'parent' symlink – need explicit dirs")
    return (child_dir / link.readlink()).resolve()


def main():
    ap = argparse.ArgumentParser(description="Diff two CRIU RAM dumps")
    ap.add_argument("paths", nargs="+", type=Path, help="CHILD or PARENT CHILD")
    ns = ap.parse_args()

    if len(ns.paths) == 1:
        child = ns.paths[0].resolve()
        parent = resolve_parent(child)
    elif len(ns.paths) == 2:
        parent, child = (p.resolve() for p in ns.paths)
    else:
        ap.error("expect CHILD  or  PARENT CHILD")

    p_idx, p_buf, p_file = build_index(parent)
    c_idx, c_buf, c_file = build_index(child)

    zero = bytes(PAGE)
    bytes_diff = pages_comp = 0

    # union of all addresses stored in either dump
    for addr in p_idx.keys() | c_idx.keys():
        pa = (
            memoryview(p_buf)[p_idx[addr] : p_idx[addr] + PAGE]
            if addr in p_idx
            else zero
        )
        ca = (
            memoryview(c_buf)[c_idx[addr] : c_idx[addr] + PAGE]
            if addr in c_idx
            else zero
        )
        if pa is ca:  # both zero_page
            continue
        d = page_diff(pa, ca)
        if d:
            bytes_diff += d
            pages_comp += 1

    print(f"bytes_diff={bytes_diff}   pages_compared={pages_comp}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"[ERR] crit decode failed: {e}")
