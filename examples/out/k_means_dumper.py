import persist
import numpy as np


def empty_list_of_ndarray() -> list[np.ndarray]:
    return []


def empty_list_of_lists(k) -> list[list[int]]:
    return [[] for _ in range(k)]


def empty_list_of_tuples() -> list[tuple[int, int]]:
    return []


def k_means(X: np.ndarray, k: int, max_iterations: int) -> np.ndarray:
    """A simple clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to the center
    of the new formed clusters. Do K-Means clustering and return cluster indices
    @param X: np.ndarray
        The dataset to cluster, where each row is a sample and each column is a feature.
    @param k: int
        The number of clusters the algorithm will form.
    @param max_iterations: int
        The number of iterations the algorithm will run for if it does
        not converge before that.
    """
    samples, features = X.shape
    centroids = np.array(X[np.random.choice(samples, k)])
    with persist.SimpleTcpClient("k_means") as transaction:
        for i in transaction.iterate(range(max_iterations)):  # type: int
            centroid_is = empty_list_of_tuples()
            for sample_i, sample in enumerate(X):
                centroid_i: int = np.argmin(np.linalg.norm(sample - centroids, axis=1))
                centroid_is.append((centroid_i, sample_i))
            clusters = empty_list_of_lists(k)
            for centroid_i, sample_i in centroid_is:
                clusters[centroid_i].append(sample_i)
            prev_centroids = centroids
            res = empty_list_of_ndarray()
            for j in range(len(clusters)):
                res.append(np.mean(X[clusters[j]], axis=0))
            centroids = np.array(res)
            diff = centroids - prev_centroids
            if not diff.any():
                break
            transaction.commit()
    y_pred = np.zeros(samples)
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred


def compute_random(n_samples: int, k: int, plot: bool) -> None:
    import sklearn.datasets

    X, y = sklearn.datasets.make_blobs(
        n_samples=n_samples, n_features=2, centers=k, cluster_std=1.8, shuffle=True
    )
    y_pred = k_means(X, k=k, max_iterations=1000)
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
