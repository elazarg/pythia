import os
import typing
from typing import Any

import pickle
import pathlib
import hashlib
import socket
import struct


FUEL_ENV = "INSTRUMENT_FUEL"


class Loader:
    fuel: int = int(os.environ.get(FUEL_ENV) or 10**6)

    restored_state: tuple[Any, ...]
    iterator: typing.Optional[typing.Iterable]
    version: int
    filename: pathlib.Path

    def __init__(self, module_filename: str | pathlib.Path, env) -> None:
        module_filename = pathlib.Path(module_filename)

        # make sure the cache is invalidated when the module changes
        h = compute_hash(module_filename, env)

        self.filename = pathlib.Path(
            f"experiment/{module_filename.parent.name}/cache/{h}.pickle"
        )
        self.csv_filename = pathlib.Path(
            f"experiment/{module_filename.parent.name}/cache/times.csv"
        )
        self.fuel = self.fuel
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.iterator = None
        self.version = 0
        self.restored_state = ()

    def __enter__(self) -> "Loader":
        if self._now_recovering():
            print("Recovering from snapshot")
            with self.filename.open("rb") as snapshot:
                self.version, self.restored_state, self.iterator = pickle.load(snapshot)
            print(f"Loaded {self.version=}: {self.restored_state}, {self.iterator}")
        with self.csv_filename.open("w"):
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            print("Finished successfully")
            self.filename.unlink()

    def iterate(self, iterable) -> typing.Iterable:
        if self.iterator is None:
            self.iterator = iter(iterable)
        return self.iterator

    def commit(self, *args) -> None:
        self.fuel -= 1
        if self.fuel <= 0:
            raise KeyboardInterrupt("Out of fuel")

        self.version += 1

        temp_filename = self.filename.with_suffix(".tmp")
        with open(temp_filename, "wb") as snapshot:
            pickle.dump((self.version, args, self.iterator), snapshot)

        pathlib.Path(self.filename).unlink(missing_ok=True)
        pathlib.Path(temp_filename).rename(self.filename)

        with open(self.csv_filename, "a") as f:
            size = pickle.dumps((self.version, args, self.iterator)).__sizeof__()
            print(self.version, size, repr(args), end="\n", flush=True, file=f)

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
        size = pickle.dumps((self.version, args, self.iterator)).__sizeof__()
        print(self.version, size, end="\n", flush=True)
        self.version += 1

    def __enter__(self) -> Loader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            print("Finished successfully")


def connect(tag: str) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("10.0.2.2", 1234))
    s.send(tag.encode("utf8"))
    return s


class SimpleTcpClient:
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
