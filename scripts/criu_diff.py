#!/usr/bin/env python3
"""
criu_diff.py – calculate the exact number of bytes that differ between an
incremental CRIU dump (*--track-mem*, **no --auto-dedup**) and its parent.

Usage
-----
    python criu_diff.py /path/to/child_dump
"""

import argparse
import json
import mmap
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")  # 4096 on x86-64


def _only(d: Path, glob: str, kind: str) -> Path:
    try:
        return next(d.glob(glob))
    except StopIteration:  # pragma: no cover
        sys.exit(f"[ERR] {d}: no {kind} ({glob}) found")


def _pagemap_for_pages(dump: Path) -> Path:
    """
    Return the pagemap whose header record has  {"pages_id": 1},
    i.e. the map that belongs to *pages-1.img* in a single-process dump.
    """
    for pm in dump.glob("pagemap-*.img"):
        hdr = json.loads(
            subprocess.check_output(
                ["python", "-m", "crit", "decode", "-i", pm], text=True
            )
        )["entries"][0]
        if hdr.get("pages_id") == 1:
            return pm
    sys.exit(f"[ERR] {dump}: no pagemap with pages_id == 1")


def _decode(pm: Path):
    """
    Yield (vaddr, nr_pages, in_parent) per entry.
    Bit 0 of *flags* == 1  ⇒  page identical to parent.
    """
    for rec in json.loads(
        subprocess.check_output(["python", "-m", "crit", "decode", "-i", pm], text=True)
    )["entries"]:
        if "vaddr" in rec:  # skip header
            yield rec["vaddr"], rec["nr_pages"], bool(rec["flags"] & 1)


@contextmanager
def build_index(dump: Path):
    """
    Yields *(index, mmap_object)* and automatically closes resources.

    *index* maps virtual address → offset inside *pages-1.img*,
    **only** for pages physically present in this dump
    (`flags & 1` == 0 in the pagemap).
    """
    pages = _only(dump, "pages-*.img", "pages image")
    pagemap = _pagemap_for_pages(dump)

    print(f"[INFO] {dump.name:<6}: {pages.name} + {pagemap.name}")

    with open(pages, "rb") as fh, mmap.mmap(
        fh.fileno(), 0, access=mmap.ACCESS_READ
    ) as buf:

        index: dict[int, int] = {}
        offset = 0
        for addr, n, in_parent in _decode(pagemap):
            for _ in range(n):
                if not in_parent:
                    index[addr] = offset
                addr += PAGE
                offset += PAGE
        yield index, buf


def _delta(a: memoryview | bytes, b: memoryview | bytes) -> int:
    """Return the number of differing bytes between two equal-length buffers."""
    return sum(x != y for x, y in zip(a, b))


def diff_dumps(child_dir: Path) -> None:
    child_dir = child_dir.resolve()
    parent_dir = (child_dir / "parent").resolve(strict=True)

    zero = bytes(PAGE)
    bytes_diff = pages_comp = 0

    with build_index(parent_dir) as (p_idx, p_buf), build_index(child_dir) as (
        c_idx,
        c_buf,
    ):
        for addr, off_child in c_idx.items():  # only pages child wrote
            child_pg = memoryview(c_buf)[off_child : off_child + PAGE]
            parent_pg = (
                memoryview(p_buf)[p_idx[addr] : p_idx[addr] + PAGE]
                if addr in p_idx
                else zero
            )
            d = _delta(child_pg, parent_pg)
            if d:
                bytes_diff += d
                pages_comp += 1
        del (
            child_pg,
            parent_pg,
        )  # memoryview objects hold references to the mmap object which should be closed

    print(f"bytes_diff={bytes_diff}   pages_compared={pages_comp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diff CRIU RAM dumps")
    parser.add_argument(
        "child_dump", type=Path, help="directory of the newer (--track-mem) dump"
    )
    args = parser.parse_args()
    try:
        diff_dumps(args.child_dump)
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        sys.exit(f"[ERR] {err}")
