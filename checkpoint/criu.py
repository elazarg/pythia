import contextlib
from os import PathLike
import os
from typing import Iterator

from checkpoint.ctypes_modern import Clibrary

criu = Clibrary("criu", "criu_")


@criu.function
def init_opts() -> int: ...


@criu.function
def check() -> int: ...


@criu.function
def dump() -> int: ...


@criu.function
def restore() -> int: ...


@criu.function
def restore_child() -> int: ...


@criu.function
def set_images_dir_fd(fd: int) -> None: ...


@contextlib.contextmanager
def set_images_dir(
    path: str | bytes | PathLike[str] | PathLike[bytes],
) -> Iterator[None]:
    fd = os.open(path, os.O_DIRECTORY)
    set_images_dir_fd(fd)
    yield
    os.close(fd)


@criu.function
def set_log_file(log_file: bytes) -> None: ...


@criu.function
def set_log_level(log_level: int) -> None: ...


@criu.function
def set_pid(pid: int) -> None: ...


@criu.function
def set_leave_running(leave_running: bool) -> None: ...


@criu.function
def set_service_address(address: bytes) -> None: ...


@criu.function
def set_track_mem(track_mem: bool) -> None: ...


@criu.function
def set_parent_images(path: bytes) -> None: ...


@criu.function
def set_ext_unix_sk(ext_unix_sk: bool) -> None: ...


@criu.function
def set_tcp_established(tcp_established: bool) -> None: ...


@criu.function
def set_evasive_devices(evasive_devices: bool) -> None: ...


@criu.function
def set_shell_job(shell_job: bool) -> None: ...


@criu.function
def set_file_locks(file_locks: bool) -> None: ...


@criu.function
def set_log_level(log_level: int) -> None: ...


@criu.function
def set_log_file(log_file: bytes) -> None: ...


@criu.function
def set_auto_dedup(auto_dedup: bool) -> None: ...


@criu.function
def set_manage_cgroups(manage: bool) -> None: ...
