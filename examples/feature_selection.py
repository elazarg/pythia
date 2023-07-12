import numpy as np
import argparse


def new(f): return f


def get_float(array: np.ndarray, idx: float) -> float:
    return array[idx]


@new
def get_ndarray(array: np.ndarray, idx: int) -> np.ndarray:
    return array[idx]


@new
def zip_arrays(left: np.ndarray, right: np.ndarray) -> list[tuple[float, float]]:
    return list(zip(left, right))


def log(idx: int, k: int) -> None:
    print(f'{idx} / {k}', end='\r', flush=True)


def do_work(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    # define new solution
    S = np.array([], int)

    for idx in range(k):  # type: int
        log(idx, k)
        # define and train model
        # preprocess features and target
        dims = np.unique(S[S >= 0])
        target = np.array(target).reshape(target.shape[0], -1)
        if features[:, dims].size > 0:
            # define sparse features
            X = features[:, dims]
            if X.ndim == 1:
                X = X.reshape(X.shape[0], 1)
            target = np.concatenate(target)
            X = (X - X.mean()) / X.std()
            X = np.c_[np.ones(X.shape[0]), X]
            theta = np.zeros(X.shape[1])
            for _ in range(10000):
                m = target.size
                error = np.dot(X, theta.T) - target
                theta = theta - (0.1 * (1 / m) * np.dot(X.T, error))
            prediction = np.zeros((len(X), 1))
            for j in range(len(X)):
                total = np.zeros(theta.shape[0])
                xj = get_ndarray(X, j)
                for i in range(len(xj)):
                    x = get_ndarray(xj, i)
                    t = get_ndarray(theta, i)
                    total += x * t
                prediction[j] = total
        else:
            prediction = np.zeros(features.shape[0]).reshape(features.shape[0], -1)
        grad = np.dot(features.T, target - prediction)

        # define vals
        A = np.array(range(len(grad)))
        points = np.setdiff1d(A, S)

        # get feasible points
        # break if points are no longer feasible
        if len(points) == 0:
            break

        # otherwise add maximum point to current solution
        a = points[0]
        for i in points:
            if get_float(grad, i) > get_float(grad, a):
                a = i

        if get_float(grad, a) >= 0:
            S = np.unique(np.append(S, a))
        else:
            break
    return S


def main(dataset: str, k) -> None:
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
