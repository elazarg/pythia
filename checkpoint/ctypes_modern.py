import ctypes
from ctypes.util import find_library
from typing import Any, Callable, Type, get_type_hints


def _get_ctype(type_hint: Type) -> Any:
    if type_hint is int:
        return ctypes.c_int
    elif type_hint is float:
        return ctypes.c_float
    elif type_hint is bool:
        return ctypes.c_bool
    elif type_hint is bytes:
        return ctypes.c_char_p
    elif type_hint is None:
        return None
    else:
        raise TypeError(f"Unsupported type hint: {type_hint}")


def structure[T](cls: Type[T]) -> Type:
    """
    A decorator to create a ctypes Structure class using type hints.
    """
    return type(
        cls.__name__,
        (ctypes.Structure,),
        {
            "_fields_": [
                (name, _get_ctype(type_hint))
                for name, type_hint in get_type_hints(cls).items()
            ]
        },
    )


class Clibrary:
    """
    A class for wrapping C libraries and their functions.
    """

    def __init__(self, library_name: str, namespace_prefix: str = "") -> None:
        self.lib = ctypes.CDLL(find_library(library_name))
        self.namespace_prefix = namespace_prefix

    def function[
        *Args, Res
    ](self, func: Callable[[*Args], Res]) -> Callable[[*Args], Res]:
        """
        A decorator method for wrapping individual C functions.
        """
        func_name = f"{self.namespace_prefix}{func.__name__}"
        annotations = func.__annotations__

        c_func = getattr(self.lib, func_name)
        c_func.restype = _get_ctype(annotations.get("return"))
        c_func.argtypes = [
            _get_ctype(arg_type)
            for arg_name, arg_type in annotations.items()
            if arg_name != "return"
        ]

        return c_func


criu = Clibrary("criu")


@criu.function
def criu_init_opts() -> int: ...


criu_init_opts()
