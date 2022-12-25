import numpy as np

def iterate(x: int):
    for i in range(x): pass


def access(x: np.ndarray):
    y = x[0]
    z = x[1:]
    m = x[x]


def tup():
    a, b = (1, 'x')

def listing():
    lst = [1]
    tpl = (1,)


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
