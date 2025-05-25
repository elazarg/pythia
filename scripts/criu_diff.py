#!/usr/bin/env python3
"""
criu_diff.py – byte-accurate delta reporter for a CRIU dumpset
(with --track-mem, no --auto-dedup, single-process).

Run on a *parent directory* that contains numbered sub-dumps (0,1,2…)
or on a single dump directory.

Example
-------
    python criu_diff.py /path/to/dumpset
    python criu_diff.py /path/to/dumpset/24
"""
import argparse
import json
import mmap
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

PAGE = os.sysconf("SC_PAGE_SIZE")


def _crit_decode(path: Path) -> dict:
    return json.loads(
        subprocess.check_output(
            ["python", "-m", "crit", "decode", "-i", path], text=True
        )
    )


def _find_pagemap(dump: Path) -> Path:
    for pm in dump.glob("pagemap-*.img"):
        if _crit_decode(pm)["entries"][0].get("pages_id") == 1:
            return pm
    sys.exit(f"[ERR] {dump}: no pagemap with pages_id==1")


def _iter_entries(pm: Path):
    for rec in _crit_decode(pm)["entries"]:
        if "vaddr" in rec:  # skip header
            assert (
                rec["nr_pages"] >= 1
            ), f"{pm}: record with non-positive nr_pages={rec['nr_pages']}"
            yield rec["vaddr"], rec["nr_pages"], bool(rec["flags"] & 1)


@contextmanager
def _index(dump: Path):
    pages = dump / "pages-1.img"
    pm = _find_pagemap(dump)

    with open(pages, "rb") as fh:
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    idx: dict[int, int] = {}
    off = stored = phantom = should_exist = 0

    for vaddr, n, in_parent in _iter_entries(pm):
        for _ in range(n):
            if not in_parent:  # CRIU says body is here
                should_exist += 1
                if off + PAGE <= buf.size():  # body really in pages-1.img
                    idx[vaddr] = off
                    off += PAGE
                    stored += 1
                else:  # body missing -> phantom
                    phantom += 1
            vaddr += PAGE  # next virtual page

    assert stored + phantom == should_exist, (
        f"{dump}: pagemap wants {should_exist} stored pages, "
        f"found bodies for {stored}, phantom={phantom}"
    )

    meta = {
        "pages_size": buf.size(),
        "stored_pages": stored,
        "phantom": phantom,
    }
    try:
        yield idx, buf, meta
    finally:
        pass  # leave mmap open; OS cleans up at process exit


def _diff(a: memoryview, b: memoryview) -> int:
    return sum(x != y for x, y in zip(a, b))


def diff_one(child: Path) -> int:
    parent = (child / "parent").resolve(strict=True)

    zero = bytes(PAGE)

    with _index(parent) as (p_idx, p_buf, p_meta):
        with _index(child) as (c_idx, c_buf, c_meta):
            bytes_diff = pages_diff = 0
            for addr, off_c in c_idx.items():
                child_pg = memoryview(c_buf)[off_c : off_c + PAGE]
                parent_pg = (
                    memoryview(p_buf)[p_idx[addr] : p_idx[addr] + PAGE]
                    if addr in p_idx
                    else zero
                )
                delta = _diff(child_pg, parent_pg)
                if delta:
                    bytes_diff += delta
                    pages_diff += 1

            identical_pages = c_meta["stored_pages"] - pages_diff

            if not bytes_diff:
                print(f"{child.name:>6}: ZERO diff")
                print(
                    f"{child.name:>6} | "
                    f"dirty={c_meta['stored_pages']:>6} "
                    f"identical={identical_pages:>6} "
                    f"diff_pages={pages_diff:>6} "
                    f"bytes_diff={bytes_diff:>10} | "
                    f"pages.img={c_meta['pages_size'] // 1024:>7} KiB "
                    f"pm_entries={c_meta['pm_entries']:>6} "
                    f"phantom={c_meta['phantom']:>6}"
                )
            del p_buf, c_buf, p_idx, c_idx, p_meta, c_meta
            return bytes_diff


def run(root: Path) -> None:
    root = root.resolve()
    if (root / "pages-1.img").exists():
        # single dump dir
        bytes_diff = diff_one(root)
        print(f"{root.name:>6}: {bytes_diff}")
        return

    dumps = sorted(
        [d for d in root.iterdir() if d.name.isdigit()], key=lambda d: int(d.name)
    )
    if not dumps:
        sys.exit(f"[ERR] {root}: no numbered dump subfolders")

    if dumps[0].name != "0":
        sys.exit(f"[ERR] expected first dump to be '0', found {dumps[0].name}")

    for d in dumps[1:]:  # skip baseline 0
        bytes_diff = diff_one(d)
        assert (
            bytes_diff >= 0
        ), f"{d}: diff_one returned negative bytes_diff={bytes_diff}"
        print(f"{d.name:>6}: {bytes_diff}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Debuggable CRIU dump differ")
    ap.add_argument(
        "dump_dir",
        type=Path,
        help="single dump dir OR parent directory of numbered dumps",
    )
    run(ap.parse_args().dump_dir)
