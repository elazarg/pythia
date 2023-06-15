import numpy as np


def empty(y: list[int]) -> list[int]:
    x = []
    for i in y:
        x.append(i)
    return x

def iterate(x: int):
    for i in range(x): pass


def cmp(a: float, b: float) -> bool:
    return a > b


def negative(b: bool) -> bool:
    return not b


def access(x: np.ndarray):
    y = x[0]
    z = x[1:]
    m = x[x]


def tup() -> tuple[int, float]:
    a, b = (1, 'x')

def listing():
    lst = [1]
    tpl = (1,)

def make_int() -> int:
    return 1

def empty_list_add(x: int):
    return [] + [x]

def list_append(y: int):
    x = []
    x.append(y)

def list_add():
    x = [1] + [make_int()]
    y = [(1,)] + [(2,)]

def pair() -> tuple[int, float]:
    x = 2
    res = x + 1
    return res

def destruct():
    a, b = pair()

def test_tuple(a: int, b: float) -> int:
    x = (a, b)
    y = x[1]
    return y
