import numpy as np
import argparse


def new(f): return f


def append_int(a: np.ndarray, n: int) -> np.ndarray:
    return np.append(a, n)


def get_float(array: np.ndarray, idx: int) -> float:
    res = array[idx]
    # assert isinstance(res, float)
    return res


def log(idx: int, k: int) -> None:
    print(f'{idx} / {k}', end='\r', flush=True)


def do_work(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    # define new solution
    # features: n x m
    # target: n x 1
    S = np.array([], "int")

    for idx in range(k):  # type: int
        log(idx, k)
        # define and train model
        # preprocess features and target
        dims = np.unique(S[S >= 0])
        target = np.array(target).reshape(target.shape[0], -1)
        X = features[:, dims]  # shape: n x |dims|
        if X.size == 0:
            prediction = np.zeros(features.shape[0]).reshape(features.shape[0], -1)
        else:
            # define sparse features
            if X.ndim == 1:
                X = X.reshape(X.shape[0], 1)
            y = np.concatenate(target)
            X = (X - X.mean()) / X.std()
            X = np.c_[np.ones(X.shape[0]), X]
            theta = np.zeros(X.shape[1])
            for _ in range(10000):
                error = np.dot(X, theta.T) - y
                theta -= 0.1 * (1 / y.size) * np.dot(X.T, error)
            prediction = np.zeros((len(X), 1))
            for j in range(len(X)):
                total = 0.0
                xj = X[j, :]
                for i in range(len(xj)):
                    x = get_float(xj, i)
                    t = get_float(theta, i)
                    total += x * t
                prediction[j] = total
        grad = np.dot(features.T, target - prediction)

        # define vals
        points = np.setdiff1d(np.array(range(len(grad))), S).astype("int")

        # get feasible points
        # break if points are no longer feasible
        if len(points) == 0:
            break

        # otherwise add maximum point to current solution
        a = points[0]
        m = get_float(grad, a)
        for i in range(len(points)):
            p = points[i]
            n = get_float(grad, p)
            if n > m:
                a = p
                m = n

        if m >= 0:
            S = np.unique(append_int(S, a))
        else:
            break
    return S


def main(dataset: str, k: int) -> None:
    features = np.load(f'data/{dataset}_features.npy')
    target = np.load(f'data/{dataset}_target.npy')
    S = do_work(features, target, k)
    print(S)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['dataset_20KB', 'dataset_large', 'healthstudy'], help='dataset to use')
    parser.add_argument('--k', type=int, default=100000, help='number of features to select')
    args = parser.parse_args()
    main(args.dataset, args.k)
