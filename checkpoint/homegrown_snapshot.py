"""
Minimal Python bindings for memory snapshot library.
Focus: zero overhead capture() and simplicity.
"""

import ctypes
from contextlib import contextmanager
from functools import partial
from typing import Callable, Iterator, Optional
import pathlib

# Load the library from this file's directory
lib_path = pathlib.Path(__file__).parent.parent / "scripts" / "snapshot.so"
_lib = ctypes.CDLL(str(lib_path))

# Setup signatures
_lib.snapshot_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lib.snapshot_init.restype = None
_lib.snapshot_capture.argtypes = [ctypes.c_void_p]
_lib.snapshot_capture.restype = None
_lib.snapshot_cleanup.argtypes = [ctypes.c_void_p]
_lib.snapshot_cleanup.restype = None


sanity_check_allocation: Optional[bytearray] = None
SANITY_CHECK_INTERVAL = 20
SANITY_CHECK_SIZE_BYTES = 1024


def make_snapshotter(step: int = 1) -> Callable[[], None]:
    if step == 1:

        @contextmanager
        def snapshotter() -> Iterator[Callable[[], None]]:
            """
            Initialize snapshotter. Returns capture function for hot loop.

            Returns:
                capture: Function that returns the number of bytes changed
            """
            # Initialize context
            _ctx = ctypes.c_void_p()
            _lib.snapshot_init(ctypes.byref(_ctx))

            # Return optimized capture function
            yield partial(_lib.snapshot_capture, _ctx)

            _lib.snapshot_cleanup(_ctx)

        return snapshotter
    else:
        iterations = 0

        @contextmanager
        def _snapshotter() -> Iterator[Callable[[], None]]:
            # Initialize context
            _ctx = ctypes.c_void_p()
            _lib.snapshot_init(ctypes.byref(_ctx))

            def capture() -> None:
                nonlocal iterations
                global sanity_check_allocation
                iterations += 1
                if (iterations % step) in (0, 1):
                    if SANITY_CHECK_INTERVAL != 0:
                        sanity_check_allocation = bytearray(SANITY_CHECK_SIZE_BYTES)
                        assert SANITY_CHECK_INTERVAL != 256
                        for i in range(SANITY_CHECK_SIZE_BYTES):
                            sanity_check_allocation[i] = iterations % 256

                    _lib.snapshot_capture(_ctx)

                    if SANITY_CHECK_INTERVAL != 0:
                        sanity_check_allocation = None

            yield capture

            _lib.snapshot_cleanup(_ctx)

        return _snapshotter
