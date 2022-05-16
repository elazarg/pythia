from __future__ import annotations

import builtins
import marshal
import struct


# Based on https://stackoverflow.com/a/67428655/2289509
def read_pyc_file(path: str) -> builtins.code:
    """Read the contents of a pyc-file."""
    with open(path, 'rb') as file:
        _magic = file.read(4)
        _size = struct.unpack('I', file.read(4))[0]
        _other = file.read(8)  # skip timestamp+size / hash string
        code = marshal.load(file)
        return code


def do_compile(file_path):
    with open(file_path, 'rb') as file:
        bytecode = file.read()

    return compile(bytecode, file_path, 'exec', dont_inherit=True, flags=0, optimize=0)


def read_function_from_file(file_path, function_name):
    """Read a function from a file."""
    code = do_compile(file_path)

    for const in code.co_consts:
        if hasattr(const, 'co_code'):
            if const.co_name == function_name:
                return const
    raise ValueError(f'Could not find function {function_name} in {file_path}')
