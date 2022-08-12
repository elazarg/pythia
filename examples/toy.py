import persist
import numpy as np


def main():
    res = np.zeros((10, 10))

    for i in range(1000):
        persist.start(i)

        temp = np.random.rand(10, 10)
        res = res + temp

        persist.commit()

    return res
