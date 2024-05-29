import numpy as np


def new(f):
    return f


@new
def empty_list_of_ints() -> list[int]:
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

    nsamples, features = X.shape
    centroids = X[np.random.choice(nsamples, k)]
    clusters: list[list[int]] = []
    # Iterate until convergence or for max iterations
    for i in range(max_iterations):  # type: int
        # print(f"{max_iterations}/{i}", end="\r", flush=True)
        # Assign samples to the closest centroids (create clusters)
        centroid_is = []
        for sample_i in range(len(X)):
            centroid_i = np.argmin(np.linalg.norm(X[sample_i] - centroids, axis=1))
            centroid_is.append((centroid_i, sample_i))
        clusters = []
        for _ in range(k):
            clusters.append(empty_list_of_ints())
        for centroid_i, sample_i in centroid_is:
            clusters[centroid_i].append(sample_i)

        # Save current centroids for convergence check
        prev_centroids = centroids
        # Calculate new centroids from the clusters
        res = []
        for j in range(len(clusters)):
            res.append(np.mean(X[clusters[j]], axis=0))
        centroids = np.array(res)
        # If no centroids have changed => convergence
        diff = centroids - prev_centroids
        if not diff.any():
            break

    y_pred = np.zeros(nsamples)
    for cluster_i in range(len(clusters)):
        for sample_i in clusters[cluster_i]:
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
