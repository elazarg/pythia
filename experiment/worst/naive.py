from checkpoint import persist
import numpy as np


def run(k: int) -> None:
    """Worst-case example"""
    [X, i] = (None,) * 2
    X = np.zeros((k,))
    with persist.Loader(__file__, locals()) as transaction:
        if transaction:
            [X, i] = transaction.move()
        for i in transaction.iterate(range(100)):  # type: int
            X[i] = 0.0
            transaction.commit(X, i)


if __name__ == "__main__":
    run(1000000)
