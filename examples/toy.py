import numpy as np


def minimal():
    res = np.zeros((5, 7))

    # comment
    i: Persist
    for i in range(10000):  # type: int
        temp = np.random.rand((5, 7))
        res += temp
    return res


class A:
    score: float
    vector: np.ndarray


def make_a() -> A:
    return A()


def not_so_minimal():
    a = make_a()
    b = make_a()
    a.vector = np.zeros((5, 7))
    a.score = 0.0
    for i in range(1000):  # type: int
        temp = np.random.rand((5, 7))
        norm = np.sum(temp)
        if a.score < norm:
            a.vector = temp
            a.score = norm
    return a
