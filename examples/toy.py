import numpy as np

def access():
    x = np.zeros((5, 7))
    res = x.astype('int')
    return res

def minimal():
    res = np.zeros((5, 7))

    for i in range(1000):
        temp = np.random.rand((5, 7))
        res += temp
    return res


class A:
    score: float
    vector: np.ndarray


def not_so_minimal():
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
