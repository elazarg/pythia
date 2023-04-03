from __future__ import annotations as _

from typing import Any

import pickle
import pathlib
import hashlib
import socket


class Iter:
    def __init__(self, iterable):
        self.iterator = iter(iterable)

    def __iter__(self):
        return self.iterator


class Loader:
    restored_state: tuple[Any, ...]

    def __init__(self, module_filename):

        with open(module_filename, 'rb') as f:
            m = hashlib.md5()
            m.update(f.read())
            h = m.hexdigest()[:6]
            name = pathlib.Path(module_filename).stem
            print(name)
        self.filename = f'cache/{name}-{h}.pickle'
        self.iterator = None
        self.version = 0
        self.restored_state = ()

    def __enter__(self) -> Loader:
        if self._now_recovering():
            print("Recovering from snapshot")
            with open(self.filename, "rb") as snapshot:
                version, self.restored_state, self.iterator = pickle.load(snapshot)
            print(f'Loaded {version=}: {self.restored_state}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print('Finished successfully')
            # pathlib.Path(self.filename).unlink()

    def iterate(self, iterable):
        if self.iterator is None:
            self.iterator = Iter(iterable)
        return self.iterator

    def commit(self, *args):
        self.version += 1

        temp_filename = f'{self.filename}.tmp'
        with open(temp_filename, "wb") as snapshot:
            pickle.dump((self.version, args, self.iterator), snapshot)

        pathlib.Path(self.filename).unlink(missing_ok=True)
        pathlib.Path(temp_filename).rename(self.filename)

    def _now_recovering(self):
        return pathlib.Path(self.filename).exists()


class PseudoLoader(Loader):
    def commit(self, *args):
        size = pickle.dumps((self.version, args, self.iterator)).__sizeof__()
        print(self.version, size, end='\n', flush=True)
        self.version += 1

    def __enter__(self) -> Loader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            print('Finished successfully')


class SimpleTcpClient:
    restored_state = ()

    def __init__(self, tag: str) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('10.0.2.2', 1234))
        self.socket.send(tag.encode('utf8'))

    def commit(self, i: int) -> None:
        self.socket.send(struct.pack('Q', i))

    def __enter__(self) -> 'SimpleTcpClient':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.socket.close()

    def iterate(self, iterable):
        return iterable
