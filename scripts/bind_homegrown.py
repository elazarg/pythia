"""
Minimal Python bindings for memory snapshot library.
Focus: zero overhead capture() and simplicity.
"""

import ctypes
from contextlib import contextmanager
from functools import partial
from typing import Callable, Iterator

# Global state
_lib = None
_ctx = None


@contextmanager
def init_snapshotter(
    output_dir: str = "/tmp", lib_path: str = "./snapshot.so"
) -> Iterator[Callable[[], None]]:
    """
    Initialize snapshotter. Returns capture function for hot loop.

    Returns:
        capture: Function that returns number of bytes changed
    """
    global _lib, _ctx

    # Load library
    _lib = ctypes.CDLL(lib_path)

    # Setup signatures
    _lib.snapshot_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    _lib.snapshot_init.restype = ctypes.c_int
    _lib.snapshot_capture.argtypes = [ctypes.c_void_p]
    _lib.snapshot_capture.restype = ctypes.c_size_t
    _lib.snapshot_cleanup.argtypes = [ctypes.c_void_p]
    _lib.snapshot_cleanup.restype = None

    # Initialize context
    _ctx = ctypes.c_void_p()
    result = _lib.snapshot_init(ctypes.byref(_ctx), output_dir.encode("utf-8"))
    if result < 0:
        raise RuntimeError("Failed to initialize")

    # Return optimized capture function
    yield partial(_lib.snapshot_capture, _ctx)

    if _lib and _ctx:
        _lib.snapshot_cleanup(_ctx)
    _lib = _ctx = None


if __name__ == "__main__":
    # Example usage
    with init_snapshotter("/tmp", "./snapshot.so") as capture:
        # Call capture_func in a hot loop
        for _ in range(100):
            changed_bytes = capture()
            print(f"Changed bytes: {changed_bytes}")
