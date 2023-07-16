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
    def __eq__(self, __o) -> bool: ...
    def __ne__(self, __o) -> bool: ...
    def __bool__(self) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __format__(self, __format_spec: str) -> str: ...
    def __getattribute__(self, __name: str) -> Any: ...
    def __sizeof__(self) -> int: ...
    def __dir__(self) -> Iterable: ...
    def __init_subclass__(cls) -> None: ...

class int:
    def __bool__(self) -> bool: ...
    def __add__(self, other: int) -> int: ...
    def __eq__(self, __o) -> bool: ...
    def __ne__(self, __o) -> bool: ...
    def __gt__(self, other: int) -> bool: ...
    def __lt__(self, other: int) -> bool: ...
    def __floordiv__(self, other: int) -> int: ...
    def __truediv__(self, other: int) -> float: ...
    def __truediv__(self, other: float) -> float: ...
    def __mul__(self, other: int) -> int: ...
    def __pow__(self, power: int) -> int: ...

class bool:
    def __bool__(self) -> bool: ...

class float:
    def __bool__(self) -> bool: ...
    def __gt__(self, other: float) -> bool: ...
    def __gt__(self, other: int) -> bool: ...
    def __lt__(self, other: float) -> bool: ...
    def __lt__(self, other: int) -> bool: ...
    def __add__(self, other: float) -> float: ...
    def __add__(self, other: int) -> float: ...
    def __sub__(self, other: float) -> float: ...
    def __sub__(self, other: int) -> float: ...
    def __mul__(self, other: float) -> float: ...
    def __mul__(self, other: int) -> float: ...
    def __truediv__(self, other: float) -> float: ...
    def __truediv__(self, other: int) -> float: ...
    def __eq__(self, __o: object) -> bool: ...
    def __ne__(self, __o: object) -> bool: ...


class str:
    def __bool__(self) -> bool: ...
    def __add__(self, other: str) -> str: ...
    def __eq__(self, other) -> bool: ...
    def __gt__(self, other: str) -> bool: ...
    def __lt__(self, other: str) -> bool: ...
    def __getitem__(self, index: int) -> str: ...
    def __len__(self) -> int: ...
    def __contains__(self, item: str) -> bool: ...
    def __iter__(self) -> Iterable[str]: ...
    def __eq__(self, __o) -> bool: ...
    def __ne__(self, __o) -> bool: ...

    def startswith(self, prefix: str) -> bool: ...
    def endswith(self, suffix: str) -> bool: ...
    def split(self) -> list[str]: ...
    def join(self, iterable: Iterable[str]) -> str: ...
    def replace(self, old: str, new: str, count: int) -> str: ...
    def strip(self, chars: str = None) -> str: ...
    def lstrip(self, chars: str = None) -> str: ...
    def rstrip(self, chars: str = None) -> str: ...
    def capitalize(self) -> str: ...
    def title(self) -> str: ...
    def lower(self) -> str: ...
    def upper(self) -> str: ...
    def swapcase(self) -> str: ...
    def casefold(self) -> str: ...
    def center(self, width: int, fillchar: str = ' ') -> str: ...


class slice:
    def __bool__(self) -> bool: ...

class tuple(Generic[*Args]):
    def __bool__(self) -> bool: ...
    def __getitem__(self: tuple[*Args], item: N) -> Args[N]: ...
    def __init__(self: tuple[*Args], *args: Args) -> None: ...
    def __add__(self: tuple[*Args], other: tuple) -> tuple: pass
    # def __add__(self: tuple[*Args], other: tuple[*Args2]) -> tuple[*Args, *Args2]: pass

class list(Generic[T]):
    def __bool__(self) -> bool: ...
    def __getitem__(self: list[T], index: N) -> T:
        result = self[index]

    @update(list[T|Q])
    def __setitem__(self: list[T], index: N, value: Q) -> None:
        self[index] = value

    @new
    def __iter__(self: list[T]) -> SupportsNext[T]:
        result += self

    def __eq__(self, __o) -> bool: ...
    def __ne__(self, __o) -> bool: ...

    @new
    def copy(self: list[T]) -> list[T]:
        result += self

    @update(list[T])
    def clear(self: list) -> None:
        del self[:]

    @update(list[T|Q])
    def append(self: list[T], x: Q) -> None:
        self[_] = x

    @update(list[T | Q])
    def insert(self: list[T], i: int, x: Q) -> None:
        self[_] = x

    @update(list[T | Q])
    def extend(self: list[T], x: list[Q]) -> None:
        self += x

    @update(list[T])
    def remove(self: list[T], x: T) -> None: pass

    @update(list[T])
    def pop(self: list[T], i: int = -1) -> T: pass

    def index(self: list[T], x: T, start: int = 0, end: int = 0) -> int: pass
    def count(self: list[T], x: T) -> int: pass

    @update(list[T])
    def sort(self: list[T], key: object = None, reverse: bool = False) -> None:
        self += self

    @update(list[T])
    def reverse(self: list[T]) -> None:
        self += self

    @new
    def __add__(self: list[T], other: list[Q]) -> list[T|Q]: pass

class SupportsNext(Protocol[T]):
    @update(SupportsNext[T])
    def __next__(self: SupportsNext[T]) -> T: ...

class Iterable(Protocol[T]):
    def __bool__(self) -> bool: ...
    @new
    def __iter__(self: Iterable[T]) -> SupportsNext[T]: ...

class Indexable(Protocol[T]):
    @new
    def __getitem__(self: Indexable[T], slice) -> T: ...

class Iterator(Protocol[T]):
    @new
    @update(Iterator[T])
    def __next__(self: Iterator[T]) -> T: ...

class range:
    def __init__(self, stop: int) -> None: ...
    def __init__(self, start: int, stop: int) -> None: ...
    def __init__(self, start: int, stop: int, step: int) -> None: ...
    @new
    def __iter__(self: range) -> SupportsNext[int]: ...

def abs(x: int) -> int: pass
def len(x) -> int: pass
def print(x) -> None: pass
def round(x: float) -> int: pass
def min(x, y) -> int: pass
def max(x, y) -> int: pass
def sum(x) -> int: pass
def all(x) -> bool: pass
def any(x) -> bool: pass

@new
def sorted(x: list[T]) -> Iterable[T]:
    result += x

@new
def sorted(x: set[T]) -> Iterable[T]:
    result += x

def zip(x: Iterable[T], y: Iterable[Q]) -> Iterable[tuple[T, Q]]: pass

@new
def iter(xs: list[T]) -> SupportsNext[T]:
    result += xs

@update(SupportsNext[T])
def next(xs: SupportsNext[T]) -> T:
    result = xs[_]

def enumerate(xs: T) -> Iterable[tuple[int, T]]:
    pass
