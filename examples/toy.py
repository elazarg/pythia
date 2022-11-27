import numpy as np


def main():
    res = np.zeros((5, 7))

    for i in range(1000):
        temp = np.random.rand(3, 5)
        res += temp
    return res


def toy1(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    # define rounds
    rounds = 0
    rounds_ind = 0.0

    # define new solution
    S = np.array([], int)

    for idx in range(k):
        # define and train model
        # preprocess current solution
        grad = linear_regression(features, target, np.unique(S[S >= 0]))
        rounds += 1
        del grad
    return S


def toy2(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    # define rounds
    rounds = 0
    rounds_ind = 0.0

    # define new solution
    S = np.array([], int)

    for idx in range(k):
        # define and train model
        # preprocess current solution
        grad = linear_regression(features, target, np.unique(S[S >= 0]))
        rounds += 1
        # define vals
        A = np.array(range(len(grad)))
        point = []
        for a in np.setdiff1d(A, S):
            point = np.append(point, a)
        del grad, A, point, a
    return S


def toy3(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    # define rounds
    rounds = 0
    rounds_ind = 0.0

    # define new solution
    S = np.array([], int)

    for idx in persist(range(k)):
        # define and train model
        # preprocess current solution
        grad = linear_regression(features, target, np.unique(S[S >= 0]))
        rounds += 1
        # define vals
        A = np.array(range(len(grad)))
        point = []
        out = [[point, len(np.setdiff1d(A, S))]]
        out = np.array(out, dtype='object')
        rounds_ind += np.max(out[:, -1])
        del grad, A, point, out
    return S
