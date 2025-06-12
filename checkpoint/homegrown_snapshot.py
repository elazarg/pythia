"""
Minimal Python bindings for memory snapshot library.
Focus: zero overhead capture() and simplicity.
"""

import ctypes
import os
from contextlib import contextmanager
from functools import partial
from typing import Callable, Iterator
import pathlib

if os.name == "posix":
    # Load the library from this file's directory
    lib_path = pathlib.Path(__file__).parent.parent / "scripts" / "snapshot.so"
    _lib = ctypes.CDLL(str(lib_path))

    # Setup signatures
    _lib.snapshot_init.argtypes = [ctypes.c_char_p]
    _lib.snapshot_init.restype = None
    _lib.snapshot_capture.argtypes = []
    _lib.snapshot_capture.restype = None
    _lib.snapshot_cleanup.argtypes = []
    _lib.snapshot_cleanup.restype = None

    SANITY_CHECK_INTERVAL = 3
    SANITY_CHECK_SIZE_BYTES = 4098
    sanity_check_allocation = bytearray(SANITY_CHECK_SIZE_BYTES)

    def dirty_memory():
        global sanity_check_allocation
        assert SANITY_CHECK_INTERVAL != 256
        print("--- DIRTYING MEMORY NOW ---")
        for i in range(SANITY_CHECK_SIZE_BYTES):
            sanity_check_allocation[i] += 1

    def make_snapshotter(step: int = 1) -> Callable[[], None]:

        @contextmanager
        def snapshotter(tag: str) -> Iterator[Callable[[], None]]:
            """
            Initialize snapshotter. Returns capture function for hot loop.

            Returns:
                capture: Function that returns the number of bytes changed
            """
            path = pathlib.Path("homegrown_images") / tag
            path.mkdir(parents=True, exist_ok=True)
            path = str(path).encode("ascii")
            # Initialize context
            _lib.snapshot_init(path)

            capture = _capture = _lib.snapshot_capture

            if step == 1:
                if SANITY_CHECK_INTERVAL != 0:

                    def capture() -> None:
                        # dirty_memory()
                        _capture()

            else:
                iterations = 0

                def capture() -> None:
                    nonlocal iterations
                    iterations += 1
                    if (iterations % step) in (0, 1):
                        # dirty_memory()
                        _capture()

            yield capture

            _lib.snapshot_cleanup()

        return snapshotter
