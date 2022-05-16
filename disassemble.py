from __future__ import annotations

import builtins
import marshal
import dis
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectFile:
    size: int
    code: builtins.code


# Based on https://stackoverflow.com/a/67428655/2289509
def read_pyc_file(path: str) -> builtins.code:
    """Read the contents of a pyc-file."""
    with open(path, 'rb') as file:
        _magic = file.read(4)
        _size = struct.unpack('I', file.read(4))[0]
        _other = file.read(8)  # skip timestamp+size / hash string
        code = marshal.load(file)
        return code


def read_function_from_file(file_path, function_name):
    """Read a function from a file."""
    code = read_pyc_file(file_path)
    for const in code.co_consts:
        if hasattr(const, 'co_code'):
            if const.co_name == function_name:
                return const
