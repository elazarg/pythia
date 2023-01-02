import persist
import numpy as np


def main():
    res = np.zeros((5, 7))

    with persist.Loader(__file__) as transaction:
        if transaction.restored_state:
            [res] = transaction.restored_state

        for i in transaction.iterate(range(10000)):
            temp = np.ones((5, 7))
            res = res + temp

            transaction.commit(res)

    return res


if __name__ == '__main__':
    print(main())
