import os
import shutil
import signal
import subprocess
import sys
import tempfile
import typing
import pickle
import pathlib
import hashlib
import socket
import struct


FUEL = "FUEL"
STEP = "STEP"


def read_fuel():
    return int(os.environ.get(FUEL, 10**6))


def read_step():
    return int(os.environ.get(STEP, 1))


def run_instrumented_file(
    instrumented: pathlib.Path,
    args: list[str],
    fuel: int,
    step: int = 1,
    capture_stdout: bool = False,
) -> str:
    stdout = None
    if capture_stdout:
        stdout = subprocess.PIPE
    result = subprocess.run(
        [sys.executable, instrumented.as_posix()] + args,
        env=os.environ | {"PYTHONPATH": os.getcwd(), FUEL: str(fuel), STEP: str(step)},
        stdout=stdout,
    )
    if capture_stdout:
        return result.stdout.decode("utf-8")
    return ""


class Loader:
    fuel: int

    restored_state: tuple
    iterator: typing.Optional[typing.Iterable]
    i: int
    filename: pathlib.Path

    def __init__(self, module_filename: str | pathlib.Path, env) -> None:
        module_filename = pathlib.Path(module_filename)
        tag = module_filename.parent.name
        results_path = pathlib.Path("results")

        # make sure the cache is invalidated when the module changes
        h = compute_hash(module_filename, env)
        cache_dir = pathlib.Path(tempfile.gettempdir()) / f"pythia-{h}"
        print(f"Using cache directory {cache_dir}", file=sys.stderr)
        cache_dir.mkdir(parents=False, exist_ok=True)
        self.filename = cache_dir / "store.pickle"
        (results_path / tag).mkdir(parents=True, exist_ok=True)
        self.tsv_filename = (results_path / tag / module_filename.stem).with_suffix(
            ".tsv"
        )

        self.fuel = read_fuel()
        self.i = -1
        self.printing_index = 1
        self.iterator = None
        self.restored_state = ()

    def __enter__(self) -> "Loader":
        if self._now_recovering():
            print("Recovering from snapshot", file=sys.stderr)
            with self.filename.open("rb") as snapshot:
                self.i, self.restored_state, self.iterator = pickle.load(snapshot)
            print(
                f"Loaded {self.i=}: {self.restored_state}, {self.iterator}",
                file=sys.stderr,
            )
        with self.tsv_filename.open("w"):
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            print("Finished successfully", file=sys.stderr)
            self.filename.unlink()

    def iterate(self, iterable) -> typing.Iterable:
        if self.iterator is None:
            self.iterator = iter(iterable)
        return self.iterator

    def commit(self, *args) -> None:
        self.i += 1
        if self.i % STEP_VALUE in [0, 1]:
            self.fuel -= 1
            if self.fuel <= 0:
                raise KeyboardInterrupt("Out of fuel")

            temp_filename = self.filename.with_suffix(".tmp")
            with open(temp_filename, "wb") as snapshot:
                pickle.dump((self.i, args, self.iterator), snapshot)

            pathlib.Path(self.filename).unlink(missing_ok=True)
            pathlib.Path(temp_filename).rename(self.filename)

            with open(self.tsv_filename, "a") as f:
                size = pickle.dumps((self.i, args, self.iterator)).__sizeof__()
                self.printing_index += 1
                print(self.printing_index, size, sep="\t", end="\n", flush=True, file=f)

    def __bool__(self) -> bool:
        return self._now_recovering()

    def move(self):
        res = self.restored_state
        del self.restored_state
        return res

    def _now_recovering(self) -> bool:
        return pathlib.Path(self.filename).exists()


def compute_hash(module_filename: pathlib.Path, *env) -> str:
    with module_filename.open("rb") as f:
        m = hashlib.md5()
        m.update(f.read())
        m.update(repr(env).encode("utf8"))
        h = m.hexdigest()[:6]
    return h


def connect(tag: str) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("10.0.2.2", 1234))
    s.send(tag.encode("utf8"))
    return s


class SimpleTcpClient:
    def __init__(self, tag: str) -> None:
        while True:
            try:
                self.socket = connect(tag)
            except ConnectionRefusedError:
                print("Could not connect to server.", file=sys.stderr)
                print(
                    "Make sure save_snapshot.py is running on host.",
                    file=sys.stderr,
                )
                print("Press Enter to retry", file=sys.stderr)
                input()
            else:
                break
        self.i = -1

    def commit(self) -> None:
        self.i += 1
        if self.i % STEP_VALUE in [0, 1]:
            # send commend to save_snapshot server to take snapshot
            self.socket.send(struct.pack("Q", self.i))
            # wait for snapshot to start, and continue after it's done
            self.socket.recv(256)

    def __enter__(self) -> "SimpleTcpClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.socket.close()

    def iterate(self, iterable):
        for value in iterable:
            yield value


def diff_vm_snapshot(folder: pathlib.Path, i: int) -> int:
    folder = folder.as_posix()
    result = subprocess.run(
        [
            "./scripts/diff_vm_snapshot",
            f"{folder}/{i}.a.dump",
            f"{folder}/{i}.b.dump",
            "64",
            "remove",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to run diff_vm_snapshot", result)
    return int(result.stdout.strip())


def diff_coredump(folder: pathlib.Path, i: int) -> int:
    folder = folder.as_posix()
    result = subprocess.run(
        [
            "./scripts/diff_coredump",
            f"{folder}/{i}.a.dump",
            f"{folder}/{i}.b.dump",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to run diff_coredump", result)
    return int(result.stdout.strip())


def coredump(tag: str, index: int) -> None:
    pid = os.getpid()
    result = subprocess.run(["gcore" "-o" f"{tag}" f"{pid}"], shell=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to run diff_coredump", result)
    os.rename(f"{tag}.{pid}", f"{tag}/{index}.core")


def make_dumps_folder(tag: str) -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    dumps_folder = cwd / "dumps" / tag
    shutil.rmtree(dumps_folder, ignore_errors=True)
    dumps_folder.mkdir(exist_ok=False, parents=True)
    return dumps_folder


def make_results_folder(tag: str) -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    results_folder = cwd / "results" / tag
    results_folder.mkdir(exist_ok=True, parents=True)
    return results_folder


PID = os.getpid()
STEP_VALUE = read_step()


def sigint() -> None:
    os.kill(PID, signal.SIGINT)


if os.name == "posix":
    try:
        from . import criu
    except AttributeError:
        print("CRIU is not available", file=sys.stderr)
    else:
        import os, pathlib, socket, struct, sys

        # ---------- globals ------------------------------------------------
        coredump_iterations = 0  # counts *loops*
        coredump_steps = 0  # counts *dumps*
        STEP_VALUE = 10  # leave as before
        CRIU_IMAGES = pathlib.Path("criu_images").absolute()

        _sock: socket.socket | None = None  # child-side handle

        # ---------- one-time CRIU init for the daemon ----------------------
        def _criu_init_options(target_pid: int) -> None:
            if criu.init_opts() < 0:
                raise OSError("CRIU init failed")

            criu.set_ext_unix_sk(True)
            with open("/sys/fs/cgroup/user.slice/helper/cgroup.procs", "w") as f:
                f.write(str(os.getpid()))
            criu.set_manage_cgroups(True)
            criu.set_pid(target_pid)  # dump *the child*

            criu.set_log_file(b"criu.log")
            criu.set_log_level(4)
            criu.set_shell_job(True)
            criu.set_leave_running(True)
            criu.set_service_address(b"/tmp/criu_service.socket")
            criu.set_track_mem("WSL" not in os.uname().release)
            criu.set_auto_dedup(False)

        # ---------- daemon loop (runs in parent) ---------------------------
        def _daemon_loop(sock: socket.socket, tag: str, child_pid: int) -> None:
            sock.setblocking(True)
            _criu_init_options(child_pid)

            while True:
                msg = sock.recv(8)  # 8-byte step number
                if not msg:  # EOF => child exited
                    break
                step = struct.unpack("<Q", msg)[0]

                folder = CRIU_IMAGES / tag / str(step)
                folder.mkdir(parents=True, exist_ok=True)

                if step:
                    criu.set_parent_images(f"../{step-1}".encode())
                with criu.set_images_dir(folder):
                    ret = criu.dump()
                    sock.send(struct.pack("<i", ret))  # 4-byte status
            os._exit(0)

        # ---------- public API --------------------------------------------
        def self_coredump(tag: str) -> None:
            """
            Forks a daemon on the first call, then blocks for each dump request.
            """
            global coredump_iterations, coredump_steps, _sock

            if _sock is None:  # first ever call -> fork
                parent_sock, child_sock = socket.socketpair()
                pid = os.fork()
                if pid == 0:  # --- CHILD (ML loop) ---
                    parent_sock.close()
                    _sock = child_sock  # keep a handle for later
                else:  # --- PARENT (daemon) ----
                    child_sock.close()
                    _daemon_loop(parent_sock, tag, pid)  # never returns

            coredump_iterations += 1

            # -------------- CHILD continues below -------------------------
            if (coredump_iterations % STEP_VALUE) in [0, 1]:
                _sock.send(struct.pack("<Q", coredump_steps))
                if struct.unpack("<i", _sock.recv(4))[0] < 0:
                    raise OSError("CRIU dump failed")
                coredump_steps += 1
