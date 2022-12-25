T = TypeVar('T')
Args = TypeVarTuple('Args')
N = TypeVar('N', Literal[int])


class object:
    __dict__: dict
    __module__: str
    __annotations__: dict
    def __init__(self) -> None: ...
    def __setattr__(self, __name: str, __value: Any) -> None: ...
    def __delattr__(self, __name) -> None: ...
    def __eq__(self, __o: object) -> bool: ...
    def __ne__(self, __o: object) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __format__(self, __format_spec: str) -> str: ...
    def __getattribute__(self, __name: str) -> Any: ...
    def __sizeof__(self) -> int: ...
    def __dir__(self) -> Iterable: ...
    def __init_subclass__(cls) -> None: ...

class int:
    def __add__(self, other: int) -> int: ...
    def __eq__(self, other) -> bool: ...

class bool:
    def __bool__(self) -> bool: ...

class float:
    def __gt__(self, other: float) -> bool: ...
    def __lt__(self, other: float) -> bool: ...

class str: pass

class slice: pass

class tuple(Generic[*Args]):
    def __getitem__(self, item: N) -> Args[N]: ...
    def __init__(self, *args: Args) -> None: ...


class list(Generic[T]):
    def __getitem__(self, item: N) -> T: ...
    def __iter__(self) -> SupportsNext[T]: ...
    @new
    def copy(self) -> list: pass
    def clear(self) -> None: pass
    def append(self, x: T) -> None: pass
    def extend(self, x: list[T]) -> None: pass
    def insert(self, i: int, x: T) -> None: pass
    def remove(self, x: T) -> None: pass
    def pop(self, i: int = -1) -> T: pass
    def index(self, x: T, start: int = 0, end: int = 0) -> int: pass
    def count(self, x: T) -> int: pass
    def sort(self, key: object = None, reverse: bool = False) -> None: pass
    def reverse(self) -> None: pass


class rec(Generic[T]):
    def foo(self, item: T) -> rec[T]: ...


class SupportsNext(Protocol[T]):
    def __next__(self) -> T: ...


class Iterable(Protocol[T]):
    @new
    def __iter__(self) -> SupportsNext[T]: ...


class Indexable(Protocol[T]):
    @new
    def __getitem__(self, slice) -> T: ...


def range(start: int) -> Iterable[int]: ...

def abs(x: int) -> int: pass
def len(x) -> int: pass
def print(x) -> None: pass
def round(x: float) -> int: pass
def min(x, y) -> int: pass
def max(x, y) -> int: pass
def sum(x) -> int: pass
def all(x) -> bool: pass
def any(x) -> bool: pass
def sorted(x) -> list: pass


class B:
    def __init__(self) -> None: ...

