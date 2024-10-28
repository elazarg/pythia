from os import PathLike

from checkpoint.ctypes_modern import Clibrary
import os

criu = Clibrary("criu")


@criu.function
def criu_init_opts() -> int: ...


@criu.function
def criu_check() -> int: ...


@criu.function
def criu_dump() -> int: ...


@criu.function
def criu_restore() -> int: ...


@criu.function
def criu_set_images_dir_fd(fd: int) -> None: ...


@criu.function
def criu_set_log_file(log_file: bytes) -> None: ...


@criu.function
def criu_set_log_level(log_level: int) -> None: ...


@criu.function
def criu_set_pid(pid: int) -> None: ...


@criu.function
def criu_set_leave_running(leave_running: bool) -> None: ...


@criu.function
def criu_set_service_address(address: bytes) -> None: ...


@criu.function
def criu_set_track_mem(track_mem: bool) -> None: ...


@criu.function
def criu_set_ext_unix_sk(ext_unix_sk: bool) -> None: ...


@criu.function
def criu_set_tcp_established(tcp_established: bool) -> None: ...


@criu.function
def criu_set_evasive_devices(evasive_devices: bool) -> None: ...


@criu.function
def criu_set_shell_job(shell_job: bool) -> None: ...


@criu.function
def criu_set_file_locks(file_locks: bool) -> None: ...


@criu.function
def criu_set_log_level(log_level: int) -> None: ...


@criu.function
def criu_set_log_file(log_file: bytes) -> None: ...


def set_criu(folder: str | bytes | PathLike[str] | PathLike[bytes]) -> None:
    if criu_init_opts() < 0:
        raise OSError("CRIU init failed")

    try:
        fd = os.open(folder, os.O_DIRECTORY)
    except OSError as e:
        raise OSError(f"Failed to open criu_images directory: {e}")
    criu_set_images_dir_fd(fd)

    criu_set_log_file(b"criu.log")
    criu_set_log_level(4)
    criu_set_pid(os.getpid())
    criu_set_leave_running(True)
    criu_set_service_address(b"/tmp/criu_service.socket")
    criu_set_track_mem(False)


if __name__ == "__main__":
    set_criu("../scripts/criu_images")
    c = criu_check()
    if c < 0:
        print(f"Failed to check CRIU: {os.strerror(c)}")
    elif c == 0:
        print("CRIU is available")
    else:
        print("CRIU is not available")

    # Perform a checkpoint
    criu_dump()
    # move criu_images/pages-1.img to criu_images/pages_old.img
    os.rename("../scripts/criu_images/pages-1.img", "criu_images/pages_old.img")
    criu_dump()
