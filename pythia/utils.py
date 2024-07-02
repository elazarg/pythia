from __future__ import annotations as _


class Function[F]:
    """A descriptor class to store a function as an attribute.
    It wraps the function in a staticmethod object, so the self argument is not passed to the function when it is called.
    """

    def __init__(self, *, default: F) -> None:
        self.f = default

    def __get__(self, obj: Function[F], objtype=None) -> F:
        return obj.f

    def __set__(self, obj: Function[F], f: F) -> None:
        obj.f = staticmethod(f)
