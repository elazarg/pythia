from __future__ import annotations as _

import ast
import marshal
import struct
from typing import Any


# Based on https://stackoverflow.com/a/67428655/2289509
def read_pyc_file(path: str) -> Any:
    """Read the contents of a pyc-file."""
    with open(path, "rb") as file:
        _magic = file.read(4)
        _size = struct.unpack("I", file.read(4))[0]
        _other = file.read(8)  # skip timestamp+size / hash string
        code = marshal.load(file)
        return code


def do_compile(file_path: str) -> Any:
    with open(file_path, "rb") as file:
        bytecode = file.read()

    return compile(bytecode, file_path, "exec", dont_inherit=True, flags=0, optimize=0)


def read_function_using_compile(file_path: str, function_name: str) -> object:
    """Read a function from a file.
    Currently, it does not support annotations"""
    code = do_compile(file_path)

    for const in code.co_consts:
        if hasattr(const, "co_code"):
            if const.co_name == function_name:
                return const
    raise ValueError(f"Could not find function {function_name} in {file_path}")


def read_file(file_path: str, filter_for_loops=True) -> tuple[dict[str, object], Any]:
    module = ast.parse("from __future__ import annotations\n", filename=file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        source = file.read()
    code = ast.parse(source, filename=file_path)

    for node in code.body:
        if isinstance(node, ast.FunctionDef):
            if not filter_for_loops:
                module.body.append(node)
                continue
            for n in ast.walk(node):
                if isinstance(n, ast.For):
                    # if n.type_comment:
                    module.body.append(node)
                    break
    functions = compile(module, "", "exec", dont_inherit=True, flags=0, optimize=0)
    env: dict[str, object] = {"new": lambda f: f}
    # exec should be safe here, since it cannot have any side effects
    exec(functions, {}, env)
    del env["annotations"]
    del env["new"]

    globals_dict: dict[str, str] = {}
    for node in code.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                globals_dict[name.asname or name.name] = name.name
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                globals_dict[name.asname or name.name] = f"{node.module}.{name.name}"
    return env, globals_dict


if __name__ == "__main__":
    pass  # read_function_as_ast('code_examples.py', 'feature_selection')
