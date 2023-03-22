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
    # Fix: wrapping in np.array to help the analysis
    centroids = np.array(X[np.random.choice(samples, k)])

    # Iterate until convergence or for max iterations
    for _ in range(max_iterations):  # type: int
        # Assign samples to the closest centroids (create clusters)
        centroid_is = empty_list_of_tuples()
        for sample_i, sample in enumerate(X):
            centroid_i = np.argmin(np.linalg.norm(sample - centroids, axis=1))
            centroid_is.append((centroid_i, sample_i))
        clusters = empty_list_of_lists(k)
        for centroid_i, sample_i in centroid_is:
            clusters[centroid_i].append(sample_i)

        # Save current centroids for convergence check
        prev_centroids = centroids
        # Calculate new centroids from the clusters
        res = empty_list_of_ndarray()
        for i in range(len(clusters)):
            res.append(np.mean(X[clusters[i]], axis=0))
        centroids = np.array(res)
        # If no centroids have changed => convergence
        diff = centroids - prev_centroids
        if not diff.any():
            break

    y_pred = np.zeros(samples)
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred


def plot_random(k: int) -> None:
    import matplotlib.pyplot as plt
    import sklearn.datasets
    # Load dataset
    X, y = sklearn.datasets.make_blobs(
        n_samples=1500, n_features=2, centers=k, cluster_std=0.5, shuffle=True
    )
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    # plt.clf()
    # Perform K-Means clustering
    y_pred = k_means(X, k=k, max_iterations=100)
    # Plot the different clusters
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()


if __name__ == "__main__":
    plot_random(4)
