import typing
from typing import Any, TypeVar

import pickle
import pathlib
import hashlib
import socket
import struct

T = TypeVar("T")


class Iter:
    def __init__(self, iterable):
        self.iterator = iter(iterable)

    def __iter__(self):
        return self.iterator


class Loader:
    restored_state: tuple[Any, ...]
    iterator: typing.Optional[Iter]
    version: int
    filename: pathlib.Path

    def __init__(self, module_filename: str | pathlib.Path):
        module_filename = pathlib.Path(module_filename)
        with module_filename.open("rb") as f:
            m = hashlib.md5()
            m.update(f.read())
            h = m.hexdigest()[:6]
        self.filename = pathlib.Path(
            f"experiment/{module_filename.parent.name}/cache/{h}.pickle"
        )
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.iterator = None
        self.version = 0
        self.restored_state = ()

    def __enter__(self) -> "Loader":
        if self._now_recovering():
            print("Recovering from snapshot")
            with self.filename.open("rb") as snapshot:
                self.version, self.restored_state, self.iterator = pickle.load(snapshot)
            print(f"Loaded {self.version=}: {self.restored_state}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            print("Finished successfully")
            self.filename.unlink()

    def iterate(self, iterable) -> Iter:
        if self.iterator is None:
            self.iterator = Iter(iterable)
        return self.iterator

    def commit(self, *args) -> None:
        self.version += 1

        temp_filename = self.filename.with_suffix(".tmp")
        with open(temp_filename, "wb") as snapshot:
            pickle.dump((self.version, args, self.iterator), snapshot)

        pathlib.Path(self.filename).unlink(missing_ok=True)
        pathlib.Path(temp_filename).rename(self.filename)
        self.restored_state = (self.version, args, self.iterator)

    def _now_recovering(self) -> bool:
        return pathlib.Path(self.filename).exists()


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

    def iterate(self, iterable: typing.Iterable[T]) -> typing.Iterable[T]:
        for self.i in self.loader.iterate(iterable):
            yield self.i
