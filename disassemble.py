from __future__ import annotations

import builtins
import inspect
import marshal
import struct
import ast


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


def read_function_using_compile(file_path, function_name):
    """Read a function from a file.
    Currently, it does not support annotations"""
    code = do_compile(file_path)

    for const in code.co_consts:
        if hasattr(const, 'co_code'):
            if const.co_name == function_name:
                return const
    raise ValueError(f'Could not find function {function_name} in {file_path}')


def read_function(file_path, funcname=None):
    module = ast.parse("from __future__ import annotations\n", filename=file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        source = file.read()
    code = ast.parse(source, filename=file_path)

    for node in code.body:
        if isinstance(node, ast.FunctionDef):
            if funcname and node.name != funcname:
                continue
            module.body.append(node)
    functions = compile(module, '', 'exec', dont_inherit=True, flags=0, optimize=0)
    env = {}
    # exec should be safe here, since it cannot have any side effects
    exec(functions, {}, env)
    del env['annotations']

    module = ast.parse("from __future__ import annotations\n", filename=file_path)
    for node in code.body:
        if False and isinstance(node, (ast.Import, ast.ImportFrom)):
            module.body.append(node)
        if isinstance(node, ast.FunctionDef):
            module.body.append(node)
    imports = compile(module, '', 'exec', dont_inherit=True, flags=0, optimize=0)

    return env, imports


if __name__ == '__main__':
    pass  # read_function_as_ast('code_examples.py', 'feature_selection')
