from checkpoint import persist
import numpy as np


def run(k: int) -> None:
    """Worst-case example"""
    X = np.zeros((k,))
    with persist.Loader(__file__, locals()) as transaction:
        if transaction:
            [X] = transaction.move()
        for i in transaction.iterate(range(100)):  # type: int
            X[i] = 0.0
            transaction.commit(X)


if __name__ == "__main__":
    run(1000000)
