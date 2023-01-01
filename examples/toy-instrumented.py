import persist
import numpy as np


def main():
    res = np.zeros((10, 10))

    locals().update(persist.load('toy-instrumented.pkl'))
    for i in persist.persistent_iteration(range(1000)):
        res = persist.start(i)

        temp = np.random.rand(10, 10)
        res = res + temp

        persist.commit(res=res)

    return res
