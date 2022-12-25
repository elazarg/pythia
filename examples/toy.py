import numpy as np

def iterate(x: int):
    for i in range(x): pass

def minimal():
    res = np.zeros((5, 7))

    for i in range(1000):
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
    a.vector = np.zeros((5, 7))
    a.score = 0.0
    for i in range(1000):
        temp = np.random.rand((5, 7))
        norm = np.sum(temp)
        if a.score < norm:
            a.vector, a.score = temp, norm
    return a
