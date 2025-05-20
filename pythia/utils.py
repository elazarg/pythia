from __future__ import annotations as _


def discard(**_) -> None:
    pass


class Function[F]:
    """A descriptor class to store a function as an attribute.
    It wraps the function in a staticmethod object, so the self argument is not passed to the function when it is called.
    """

    def __init__(self, *, default: F) -> None:
        # Store the function as a staticmethod from the beginning.
        self.f: staticmethod = staticmethod(default)

    def __get__(self, instance, owner: type | None = None) -> F:
        # self: the Function descriptor instance (e.g., Cfg.annotator)
        # instance: the instance of the class that owns this descriptor (e.g., a Cfg instance), or None if accessed via the class
        # owner: the class that owns this descriptor (e.g., Cfg class)

        # self.f is a staticmethod object. Accessing its __get__
        # returns the underlying function, which is what we want to return.
        # This ensures it's callable without an implicit 'self' or 'instance'.
        return self.f.__get__(instance, owner)  # type: ignore

    def __set__(self, obj: Function[F], f: F) -> None:
        obj.f = staticmethod(f)
