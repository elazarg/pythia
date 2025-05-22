#!/usr/bin/env python3
"""
criu_diff.py - Byte-accurate RAM delta for one CRIU checkpoint
===============================================================

Given the directory of a **child** dump created with

    criu dump --track-mem ... # NO --auto-dedup

the script:

1.  Follows the *parent* symlink in that directory.
2.  Locates `pages-1.img` **and** the matching `pagemap-*.img`
    (the one whose header record has `"pages_id": 1`).
3.  Builds an index of **only the pages actually stored** in each dump
    (pages with `flags & 1` → `IN_PARENT` are ignored).
4.  Compares every stored child page to the corresponding parent page,
    counting differing bytes.

The result is the minimum number of bytes an ideal fine-grained differ
would need to write so that **parent + delta = child RAM image**.

Usage
-----
    python criu_diff.py /path/to/child_dump

Requirements
------------
* Python ≥ 3.9 (tested on 3.12)
* `python -m crit` in `$PATH`
  (pycriu built against protobuf < 4, as you already configured)
* Dumps are single-process (one `pages-*.img`).
"""

import argparse
import json
import mmap
import os
import subprocess
import sys
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")  # 4096 on x86-64


def pick_pages_img(dump: Path) -> Path:
    try:
        return next(dump.glob("pages-*.img"))
    except StopIteration:
        sys.exit(f"[ERR] {dump}: no pages-*.img found")


def pick_pagemap_img(dump: Path) -> Path:
    """
    Return the pagemap whose header has  {"pages_id": 1},
    i.e. the map that describes pages-1.img.
    """
    for pm in dump.glob("pagemap-*.img"):
        raw = subprocess.check_output(
            ["python", "-m", "crit", "decode", "-i", str(pm)], text=True
        )
        first = json.loads(raw)["entries"][0]
        if first.get("pages_id") == 1:
            return pm
    sys.exit(f"[ERR] {dump}: no pagemap with pages_id==1 found")


def decode_pagemap(path: Path):
    """
    Yield tuples (vaddr, nr_pages, in_parent).

    Record schema (current CRIU):
        {
          "vaddr":    <u64>,
          "nr_pages": <u32>,
          "flags":    <u32>   # bit 0 == IN_PARENT
        }
    """
    data = subprocess.check_output(
        ["python", "-m", "crit", "decode", "-i", str(path)], text=True
    )
    for rec in json.loads(data)["entries"]:
        if "vaddr" not in rec:  # skip header
            continue
        yield rec["vaddr"], rec["nr_pages"], bool(rec["flags"] & 1)


def build_index(dump: Path):
    """
    Build {vaddr → offset} for every page stored in this dump
    (pages marked IN_PARENT are *not* indexed).

    Returns (index, mmap_object).  The open file object is kept
    alive by attaching it to the mmap.
    """
    pages = pick_pages_img(dump)
    pagemap = pick_pagemap_img(dump)

    print(f"[INFO] {dump.name:<6}: {pages.name} + {pagemap.name}")

    fh = open(pages, "rb")
    buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    buf._fh = fh  # keep FD alive

    index, off = {}, 0
    for vaddr, n, in_parent in decode_pagemap(pagemap):
        for _ in range(n):
            if not in_parent:
                index[vaddr] = off
            vaddr += PAGE
            off += PAGE
    return index, buf


def page_diff(a: memoryview, b: memoryview) -> int:
    """Return the number of differing bytes between two 4 KiB pages."""
    return sum(x != y for x, y in zip(a, b))


def main(child_dir: Path):
    child_dir = child_dir.resolve()

    parent_link = child_dir / "parent"
    if not parent_link.exists():
        sys.exit(
            f"[ERR] {child_dir}: missing 'parent' symlink "
            "(dump was not incremental?)"
        )
    parent_dir = (child_dir / parent_link.readlink()).resolve()

    p_idx, p_buf = build_index(parent_dir)
    c_idx, c_buf = build_index(child_dir)

    zero_page = bytes(PAGE)
    bytes_diff = pages_comp = 0

    # Compare *only* pages that the child actually wrote
    for addr, off_child in c_idx.items():
        child_page = memoryview(c_buf)[off_child : off_child + PAGE]
        if addr in p_idx:
            parent_page = memoryview(p_buf)[p_idx[addr] : p_idx[addr] + PAGE]
        else:
            parent_page = zero_page  # address absent in parent
        delta = page_diff(child_page, parent_page)
        if delta:
            bytes_diff += delta
            pages_comp += 1

    print(f"bytes_diff={bytes_diff}   pages_compared={pages_comp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diff CRIU RAM dumps")
    parser.add_argument(
        "child_dump", type=Path, help="directory of the newer (--track-mem) dump"
    )
    args = parser.parse_args()
    try:
        main(args.child_dump)
    except subprocess.CalledProcessError as err:
        sys.exit(f"[ERR] crit decode failed: {err}")
