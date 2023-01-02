
import persist
import numpy as np

def minimal():
    res = np.zeros((5, 7))
    i: Persist
    with persist.Loader(__file__) as transaction:
        if transaction.restored_state:
            [res] = transaction.restored_state
        for i in transaction.iterate(range(10000)): # type: int
            temp = np.random.rand(5, 7)
            res += temp
            transaction.commit(res)
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
    with persist.Loader(__file__) as transaction:
        if transaction.restored_state:
            [a] = transaction.restored_state
        for i in transaction.iterate(range(1000)): # type: int
            temp = np.random.rand(5, 7)
            norm = np.sum(temp)
            if a.score < norm:
                a.vector = temp
                a.score = norm
            transaction.commit(a)
    return a

if __name__ == '__main__':
    print(minimal())
