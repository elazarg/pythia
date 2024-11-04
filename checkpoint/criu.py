from os import PathLike
import os

from checkpoint.ctypes_modern import Clibrary

lib = Clibrary("criu", "criu_")


@lib.function
def init_opts() -> int: ...


@lib.function
def check() -> int: ...


@lib.function
def dump() -> int: ...


@lib.function
def restore() -> int: ...


@lib.function
def restore_child() -> int: ...


@lib.function
def set_images_dir_fd(fd: int) -> None: ...


def set_images_dir(path: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
    try:
        fd = os.open(path, os.O_DIRECTORY)
    except OSError as e:
        raise OSError(f"Failed to open criu_images directory: {e}")
    set_images_dir_fd(fd)


@lib.function
def set_log_file(log_file: bytes) -> None: ...


@lib.function
def set_log_level(log_level: int) -> None: ...


@lib.function
def set_pid(pid: int) -> None: ...


@lib.function
def set_leave_running(leave_running: bool) -> None: ...


@lib.function
def set_service_address(address: bytes) -> None: ...


@lib.function
def set_track_mem(track_mem: bool) -> None: ...


@lib.function
def set_parent_images(path: bytes) -> None: ...


@lib.function
def set_ext_unix_sk(ext_unix_sk: bool) -> None: ...


@lib.function
def set_tcp_established(tcp_established: bool) -> None: ...


@lib.function
def set_evasive_devices(evasive_devices: bool) -> None: ...


@lib.function
def set_shell_job(shell_job: bool) -> None: ...


@lib.function
def set_file_locks(file_locks: bool) -> None: ...


@lib.function
def set_log_level(log_level: int) -> None: ...


@lib.function
def set_log_file(log_file: bytes) -> None: ...


@lib.function
def set_auto_dedup(auto_dedup: bool) -> None: ...
