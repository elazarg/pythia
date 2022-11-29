import numpy as np


def minimal():
    res = np.zeros((5, 7))

    for i in range(1000):
        temp = np.random.rand(3, 5)
        res += temp
    return res


def not_so_minimal():
    # class A:
    #     score: float
    #     vector: np.ndarray
    a = A()
    a.vector = np.zeros((5, 7))
    a.score = 0.0
    for i in range(1000):
        temp = np.random.rand((5, 7))
        norm = np.sum(temp)
        if a.score < norm:
            a.vector = temp
            a.score = norm
    return a
