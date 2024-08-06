from checkpoint import persist
import numpy as np


def run(k: int) -> None:
    """Worst-case example"""
    X = np.zeros((k,))
    with persist.SimpleTcpClient("worst") as client:
        for i in client.iterate(range(100)):  # type: int
            X[i] = 0.0
            client.commit()


if __name__ == "__main__":
    run(1000000)
