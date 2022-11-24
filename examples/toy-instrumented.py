import persist
import numpy as np


def main():
    res = np.zeros((10, 10))

    locals().update(persist.load())
    for i in persist.range(1000):
        persist.start(i)

        temp = np.random.rand(10, 10)
        res = res + temp
        persist.mark('$1')

        del temp
        persist.commit()

    return res
