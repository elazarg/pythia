from __future__ import annotations as _

import ast
import marshal
import pathlib
import struct
from dataclasses import dataclass
from typing import Any

from pythia.utils import discard as _discard
from pythia import ast_transform


# Based on https://stackoverflow.com/a/67428655/2289509
def read_pyc_file(path: str) -> Any:
    """Read the contents of a pyc-file."""
    with open(path, "rb") as file:
        _discard(_magic=file.read(4))
        _discard(_size=struct.unpack("I", file.read(4))[0])
        _discard(_other=file.read(8))  # skip timestamp+size / hash string
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


@dataclass
class ParsedFile:
    functions: dict[str, object]
    imports: dict[str, str]
    annotated_for: dict[str, frozenset[int]]


def read_file(file_path: str) -> ParsedFile:
    module = ast.parse("from __future__ import annotations\n", filename=file_path)
    source = pathlib.Path(file_path).read_text(encoding="utf-8")
    parser = ast_transform.Parser(file_path)
    code = parser.parse(source)
    annotated_for: dict[str, frozenset[int]] = {}
    for funcdef in parser.iterate_purified_functions(code):
        module.body.append(funcdef)
        annotated_for[funcdef.name] = ast_transform.annotated_for_labels(funcdef)

    module_with_functions = compile(
        module, "", "exec", dont_inherit=True, flags=0, optimize=0
    )
    functions_env: dict[str, object] = {"new": lambda f: f}
    # exec should be safe here, since it cannot have any side effects
    exec(module_with_functions, {}, functions_env)
    del functions_env["annotations"]
    del functions_env["new"]
    if not functions_env:
        raise ValueError("No functions found")

    imports: dict[str, str] = {}
    for node in code.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                imports[name.asname or name.name] = name.name
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                imports[name.asname or name.name] = f"{node.module}.{name.name}"

    return ParsedFile(
        functions=functions_env,
        imports=imports,
        annotated_for=annotated_for,
    )


if __name__ == "__main__":
    print(read_file("test_data/lists.py"))
