import numpy as np


def main():
    res = np.zeros((5, 7))

    for i in range(1000):
        temp = np.random.rand(3, 5)
        res = res + temp

        del temp
    return res
