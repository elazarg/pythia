from checkpoint import persist
import numpy as np
import sklearn.datasets

np.random.seed(42)


def run(X: np.ndarray, k: int, max_iterations: int) -> np.ndarray:
    """A simple clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to the center
    of the newly formed clusters. Do K-Means clustering and return cluster indices
    @param X: np.ndarray
        The dataset to cluster, where each row is a sample and each column is a feature.
    @param k: int
        The number of clusters the algorithm will form.
    @param max_iterations: int
        The number of iterations the algorithm will run for if it does
        not converge before that.
    """
    centroids = X[np.random.choice(X.shape[0], k)]
    clusters = list[list[int]]()
    with persist.snapshotter("k_means") as self_coredump:
        for i in range(max_iterations):  # type: int
            self_coredump()
            clusters = [list[int]() for _ in range(k)]
            for sample_i in range(len(X)):
                r = np.linalg.norm(X[sample_i] - centroids, None, 1).argmin()
                clusters[r].append(sample_i)
            new_centroids = np.array([X[cluster].mean(0) for cluster in clusters])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
    y_pred = np.zeros(X.shape[0])
    for cluster_i in range(len(clusters)):
        for sample_i in clusters[cluster_i]:
            y_pred[sample_i] = cluster_i
    return y_pred


def compute_random(n_samples: int, k: int, plot: bool) -> None:
    X, y = sklearn.datasets.make_blobs(
        n_samples=n_samples, n_features=2, centers=k, cluster_std=1.8, shuffle=True
    )
    y_pred = run(X, k=k, max_iterations=1000)
    if plot:
        import matplotlib.pyplot as plt

        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.show()
    else:
        print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    compute_random(args.samples, args.k, args.plot)
