import numpy as np


def run(k: int) -> None:
    """Worst-case example"""
    X = np.zeros((k,))
    for i in range(100):  # type: int
        X[i] = 0.0


if __name__ == "__main__":
    run(1000000)
