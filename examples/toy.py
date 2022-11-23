import persist
import numpy as np


def main():
    res = np.zeros((5, 7))

    # locals().update(persist.load())
    for i in range(1000):
        persist.start(i)

        temp = np.random.rand(3, 5)
        res = res + temp

        persist.commit()
        del temp

    return res
