T = TypeVar('T')
Q = TypeVar('Q')
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
    def __gt__(self, other: int) -> bool: ...
    def __lt__(self, other: int) -> bool: ...

class bool:
    def __bool__(self) -> bool: ...

class float:
    def __gt__(self, other: float) -> bool: ...
    def __gt__(self, other: int) -> bool: ...
    def __lt__(self, other: float) -> bool: ...
    def __lt__(self, other: int) -> bool: ...

class str: pass

class slice: pass

class tuple(Generic[*Args]):
    def __getitem__(self, item: N) -> Args[N]: ...
    def __init__(self, *args: Args) -> None: ...
    def __add__(self: tuple, other: tuple) -> tuple: pass
    # def __add__(self: tuple[*Args], other: tuple[*Args2]) -> tuple[*Args, *Args2]: pass

class list(Generic[T]):
    def __getitem__(self: list[T], item: N) -> T: ...
    def __iter__(self: list[T]) -> SupportsNext[T]: ...

    @new
    def copy(self: list[T]) -> list[T]: pass

    def clear(self: list[T]) -> None: pass

    def append(self: list[T], x: T) -> None: pass

    def extend(self: list[T], x: list[T]) -> None: pass
    def insert(self: list[T], i: int, x: T) -> None: pass
    def remove(self: list[T], x: T) -> None: pass
    def pop(self: list[T], i: int = -1) -> T: pass
    def index(self: list[T], x: T, start: int = 0, end: int = 0) -> int: pass
    def count(self: list[T], x: T) -> int: pass
    def sort(self: list[T], key: object = None, reverse: bool = False) -> None: pass
    def reverse(self: list[T]) -> None: pass

    def __add__(self: list[T], other: list[Q]) -> list[T|Q]: pass

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

class Iterator(Protocol[T]):
    @new
    def __next__(self) -> T: ...

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

def enumerate(xs: T) -> Iterable[tuple[int, T]]: pass