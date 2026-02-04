from checkpoint import persist
import sys
import numpy as np
import argparse


def get_scalar[T](array: np.ndarray[T], idx: int) -> T:
    return array[idx]


def log(idx: int, k: int) -> None:
    print(f"{idx} / {k}", end="\r", flush=True, file=sys.stderr)


def run(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """select k features from features using target as the target variable"""
    S = np.array([], "int")
    with persist.snapshotter("omp") as self_coredump:
        for idx in range(k):  # type: int
            self_coredump()
            log(idx, k)
            dims = np.unique(S[S >= 0])
            target = np.array(target).reshape(target.shape[0], -1)
            X = features[:, dims]
            if X.size == 0:
                prediction = np.zeros(features.shape[0]).reshape(features.shape[0], -1)
            else:
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
                        x = get_scalar(xj, i)
                        t = get_scalar(theta, i)
                        total += x * t
                    prediction[j] = total
            grad = np.dot(features.T, target - prediction)
            points = np.setdiff1d(np.array(range(len(grad))), S).astype("int")
            if len(points) == 0:
                break
            a = points[0]
            m = get_scalar(grad, a)
            for i in range(len(points)):
                p = points[i]
                n = get_scalar(grad, p)
                if n > m:
                    a = p
                    m = n
            if m >= 0:
                S = np.unique(np.append(S, a))
            else:
                break
    return S


def main(dataset: str, k: int) -> None:
    features = np.load(f"experiment/omp/{dataset}_features.npy")
    target = np.load(f"experiment/omp/{dataset}_target.npy")
    S = run(features, target, k)
    print(S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["dataset_20KB", "dataset_large", "healthstudy"],
        help="dataset to use",
    )
    parser.add_argument(
        "--k", type=int, default=100000, help="number of features to select"
    )
    args = parser.parse_args()
    main(args.dataset, args.k)
