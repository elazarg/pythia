import os
import subprocess
import sys
import tempfile
import typing
from typing import Any

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
    instrumented: str,
    args: list[str],
    fuel: int,
    step: int = 1,
    capture_stdout: bool = False,
) -> str:
    stdout = None
    if capture_stdout:
        stdout = subprocess.PIPE
    result = subprocess.run(
        [sys.executable, instrumented] + args,
        env=os.environ | {"PYTHONPATH": os.getcwd(), FUEL: str(fuel), STEP: str(step)},
        stdout=stdout,
    )
    if capture_stdout:
        return result.stdout.decode("utf-8")
    return ""


class Loader:
    fuel: int
    step: int

    restored_state: tuple[Any, ...]
    iterator: typing.Optional[typing.Iterable]
    i: int
    filename: pathlib.Path

    def __init__(self, module_filename: str | pathlib.Path, env) -> None:
        module_filename = pathlib.Path(module_filename)

        # make sure the cache is invalidated when the module changes
        h = compute_hash(module_filename, env)
        cachedir = pathlib.Path(tempfile.gettempdir()) / f"pythia-{h}"
        print(f"Using cache directory {cachedir}", file=sys.stderr)
        cachedir.mkdir(parents=False, exist_ok=True)
        self.filename = cachedir / "store.pickle"
        self.csv_filename = cachedir / "times.csv"

        self.fuel = read_fuel()
        self.i = -1
        self.step = read_step()
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
        with self.csv_filename.open("w"):
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
        if self.i % self.step == 0:
            self.fuel -= 1
            if self.fuel <= 0:
                raise KeyboardInterrupt("Out of fuel")

            temp_filename = self.filename.with_suffix(".tmp")
            with open(temp_filename, "wb") as snapshot:
                pickle.dump((self.i, args, self.iterator), snapshot)

            pathlib.Path(self.filename).unlink(missing_ok=True)
            pathlib.Path(temp_filename).rename(self.filename)

            with open(self.csv_filename, "a") as f:
                size = pickle.dumps((self.i, args, self.iterator)).__sizeof__()
                print(self.i, size, repr(args), end="\n", flush=True, file=f)

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


class PseudoLoader(Loader):
    def commit(self, *args) -> None:
        size = pickle.dumps((self.i, args, self.iterator)).__sizeof__()
        print(self.i, size, end="\n", flush=True, file=sys.stderr)
        self.i += 1

    def __enter__(self) -> Loader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            print("Finished successfully", file=sys.stderr)


def connect(tag: str) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("10.0.2.2", 1234))
    s.send(tag.encode("utf8"))
    return s


class WrapperTcpClient:
    loader: Loader
    socket: socket.socket
    i: Any

    def __init__(self, tag: str, loader: Loader) -> None:
        self.loader = loader
        self.socket = connect(tag)
        self.i = None

    @property
    def restored_state(self) -> tuple[Any, ...]:
        return self.loader.restored_state

    def commit(self, *args) -> None:
        self.loader.commit(*args)
        data = struct.pack("Q", self.i, *self.restored_state)
        self.socket.send(data)
        self.socket.recv(256)  # wait for snapshot

    def __enter__(self) -> "SimpleTcpClient":
        self.loader = self.loader.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.loader.__exit__(exc_type, exc_val, exc_tb)
        self.socket.close()

    def iterate[T](self, iterable: typing.Iterable[T]) -> typing.Iterable[T]:
        for self.i in self.loader.iterate(iterable):
            yield self.i


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
        self.step = read_step()
        self.i = -1
        self.value = None

    def commit(self) -> None:
        self.i += 1
        if self.i % self.step in [0, 1]:
            # send commend to save_snapshot server to take snapshot
            self.socket.send(struct.pack("Q", self.i))
            # wait for snapshot to start, and continue after it's done
            self.socket.recv(256)

    def __enter__(self) -> "SimpleTcpClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.socket.close()

    def iterate(self, iterable):
        for self.value in iterable:
            yield self.value
