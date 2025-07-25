from typing import Iterable, Iterator, Literal, Any

class ellipsis: ...

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
    def __bool__(self: float) -> bool: ...
    def __gt__(self: float, other: float) -> bool: ...
    def __gt__(self: float, other: int) -> bool: ...
    def __lt__(self: float, other: float) -> bool: ...
    def __lt__(self: float, other: int) -> bool: ...
    def __add__(self: float, other: float) -> float: ...
    def __add__(self: float, other: int) -> float: ...
    def __sub__(self: float, other: float) -> float: ...
    def __sub__(self: float, other: int) -> float: ...
    def __mul__(self: float, other: float) -> float: ...
    def __mul__(self: float, other: int) -> float: ...
    def __truediv__(self: float, other: float) -> float: ...
    def __truediv__(self: float, other: int) -> float: ...
    def __eq__(self: float, __o: object) -> bool: ...
    def __ne__(self: float, __o: object) -> bool: ...

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
    def center(self, width: int, fillchar: str = " ") -> str: ...

class slice:
    def __bool__(self) -> bool: ...

class tuple[*Args]:
    def __bool__(self) -> bool: ...
    def __getitem__[N: Literal[int]](self: tuple[*Args], item: N) -> Args[N]: ...
    def __init__(self: tuple[*Args], *args: Args) -> None: ...
    def __contains__(self, object) -> bool: ...
    def __add__(self: tuple[*Args], other: tuple) -> tuple:
        pass
    # def __add__(self: tuple[*Args], other: tuple[*Args2]) -> tuple[*Args, *Args2]: pass

class list[T]:
    @update(list[T], *1)
    def __init__(self, iterable: Iterable[T] = None) -> None: ...
    def __init__(self) -> None: ...
    def __bool__(self) -> bool: ...
    def __getitem__[N: Literal[int]](self: list[T], index: N) -> T:
        result = self[index]

    @update(list[T | Q], 2)
    def __setitem__[Q, N](self: list[T], index: N, value: Q) -> None:
        self[index] = value

    @new
    def __iter__(self: list[T]) -> Iterator[T]:
        result += self

    def __contains__(self, object) -> bool: ...
    def __eq__(self, __o) -> bool: ...
    def __ne__(self, __o) -> bool: ...
    @new
    def copy(self: list[T]) -> list[T]:
        result += self

    @update(list[T])
    def clear(self: list) -> None:
        del self[:]

    @update(list[T | Q], 1)
    def append[Q](self: list[T], x: Q) -> None:
        self[_] = x

    @update(list[T | Q], 2)
    def insert[Q](self: list[T], i: int, x: Q) -> None:
        self[_] = x

    @update(list[T | Q], *2)
    def extend[Q](self: list[T], x: list[Q]) -> None:
        self += x

    @update(list[T])
    def remove(self: list[T], x: T) -> None:
        pass

    @update(list[T])
    def pop(self: list[T]) -> T:
        pass

    @update(list[T])
    def pop(self: list[T], i: int) -> T:
        pass

    def index(self: list[T], x: T, start: int = 0, end: int = 0) -> int:
        pass

    def count(self: list[T], x: T) -> int:
        pass

    @update(list[T])
    def sort(self: list[T], key: object = None, reverse: bool = False) -> None:
        self += self

    @update(list[T])
    def reverse(self: list[T]) -> None:
        self += self

    @new
    def __add__[Q](self: list[T], other: list[Q]) -> list[T | Q]:
        pass

class set[T]:
    @update(set[T], *1)
    def __init__(self, other: Iterable[T]) -> None: ...
    def __init__(self) -> None: ...
    def __bool__(self) -> bool: ...
    @new
    def __iter__(self: set[T]) -> Iterator[T]:
        result += self

    def __contains__(self, object) -> bool: ...
    def __eq__(self, __o) -> bool: ...
    def __ne__(self, __o) -> bool: ...
    @new
    def copy(self: set[T]) -> set[T]:
        result += self

    @update(set[T])  # set[Bottom]
    def clear(self: set[T]) -> None:
        del self[:]

    @update(set[T | Q], 1)
    def add[Q](self: set[T], x: Q) -> None:
        self[_] = x

    @update(set[T])
    def remove(self: set[T], x: T) -> None:
        pass

    def intersection_update[T](self: set[T], other: set[T]) -> None:
        pass

    @update(set[T])
    def pop(self: set[T]) -> T:
        pass

    @new
    def __add__[Q](self: set[T], other: set[Q]) -> set[T | Q]:
        pass

class dict[K, V]:
    def __bool__(self) -> bool: ...
    @new
    def __iter__(self: dict[K, V]) -> Iterator[K]:
        result += self

    def __contains__(self, object) -> bool: ...
    def __getitem__(self: dict[K, V], key: K) -> V: ...
    @update(dict[K | K1, V | V1], 2)
    def __setitem__[K1, V1](self: dict[K, V], key: K1, value: V1) -> None: ...
    @new
    def keys(self: dict[K, V]) -> Iterable[K]:
        result += self

    @new
    def values(self: dict[K, V]) -> Iterable[V]:
        result += self

    def __eq__(self, __o) -> bool: ...
    def __ne__(self, __o) -> bool: ...
    @new
    def copy(self: dict[K, V]) -> dict[K, V]:
        result += self

    @update(dict[K, V])
    def clear(self: dict[K, V]) -> None:
        del self[:]

class Iterator[T](Protocol):
    @update(Iterator[T])
    def __next__(self: Iterator[T]) -> T: ...

class Iterable[T](Protocol):
    @new
    def __iter__(self: Iterable[T]) -> Iterator[T]: ...

class Indexable[T](Protocol):
    @new
    def __getitem__(self: Indexable[T], slice) -> T: ...

class Iterator[T](Protocol):
    @new
    @update(Iterator[T])
    def __next__(self: Iterator[T]) -> T: ...

class range:
    def __init__(self, stop: int) -> None: ...
    def __init__(self, start: int, stop: int) -> None: ...
    def __init__(self, start: int, stop: int, step: int) -> None: ...
    @new
    def __iter__(self: range) -> Iterator[int]: ...

def abs(x: int) -> int:
    pass

def len(x) -> int:
    pass

def print(x) -> None:
    pass

def round(x: float) -> int:
    pass

def min[T](x: T, y: T) -> T:
    pass

def min[T](xs: Iterable[T]) -> T:
    pass

def max[T](x: T, y: T) -> T:
    pass

def max[T](xs: Iterable[T]) -> T:
    pass

def sum[T](xs: Iterable[T]) -> T:
    pass

def all(x) -> bool:
    pass

def any(x) -> bool:
    pass

@new
def sorted[T](x: list[T]) -> Iterable[T]:
    result += x

@new
def sorted[T](x: set[T]) -> Iterable[T]:
    result += x

def zip[T, Q](x: Iterable[T], y: Iterable[Q]) -> Iterable[tuple[T, Q]]:
    pass

@new
def iter[T](xs: list[T]) -> Iterator[T]:
    result += xs

@update(Iterator[T])
def next[T](xs: Iterator[T]) -> T:
    result = xs[_]

def enumerate[T](xs: Iterable[T]) -> Iterable[tuple[int, T]]:
    pass
