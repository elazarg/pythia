import persist
import numpy as np


def main():
    res = np.zeros((3, 3))

    transaction = persist.Loader(__file__)
    with transaction as restored_state:
        if restored_state:
            [res] = restored_state

        for i in transaction.iterate(range(10000)):
            temp = np.ones((3, 3))
            res = res + temp

            transaction.commit(res)

    return res

if __name__ == '__main__':
    print(main())
