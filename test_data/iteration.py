import numpy as np
import collections
from typing import Iterable


def new(f):
    return f


def first_shape(x: np.ndarray) -> int:
    return x.shape[0]


def setitem(x: np.ndarray, i: int, y: float) -> None:
    x[i] = y


def counter(i: int, j: int, f: np.ndarray) -> None:
    res = collections.Counter()
    res[i] += j


def length(xs: Iterable[int]) -> int:
    return max(xs)


def comprehension(xs: list[int], k: int):
    x = [k for k in xs]
    # y = [k for k in xs]
    return k


def get_world(g: dict[int, set[int]], root_to_leaf_path: list[int]) -> set[int]:
    world = set(g.keys())
    for u in root_to_leaf_path:
        world.intersection_update(g[u])
    return world


def loopfor():
    for x in range(5):
        return x


def test_dict(g: dict[int, bool]) -> set[int]:
    ks = g.keys()
    return set(ks)


def iterate(x: int):
    for i in range(x):
        pass
    x = 5
    print(x)


def double_iterate(xxs: list[list[int]], ys: list[int]):
    for xs in range(len(xxs)):
        for x in range(xs):
            ys[x] = x


def cmp(a: float, b: float) -> bool:
    return a > b


def negative(b: bool) -> bool:
    return not b


def access(x: np.ndarray):
    y = x[0]
    z = x[1:]
    m = x[x]


def tup() -> tuple[int, float]:
    return (1, "x")


def listing():
    lst = [1]
    tpl = (1,)


def make_int() -> int:
    return 1


def simple_tuple() -> tuple[int, int]:
    return (1, 2)


def destruct():
    a, b = simple_tuple()


def test_tuple_simple_assign(a: int, b: float) -> None:
    (c, d) = (a, b)


def test_tuple_assign_through_var(a: int, b: float) -> None:
    x = (a, b)
    (c, d) = x


def test_tuple2(a: int, b: float) -> int:
    x = (a, b)
    y = x[1]
    return y
