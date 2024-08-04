from checkpoint import persist
import numpy as np
import sklearn.datasets


def run(X: np.ndarray, k: int, max_iterations: int) -> np.ndarray:
    [
        X,
        _,
        centroids,
        cluster,
        cluster_i,
        clusters,
        diff,
        features,
        i,
        k,
        max_iterations,
        nsamples,
        prev_centroids,
        r,
        sample_i,
        y_pred,
    ] = (None,) * 16
    "A simple clustering method that forms k clusters by iteratively reassigning\n    samples to the closest centroids and after that moves the centroids to the center\n    of the new formed clusters. Do K-Means clustering and return cluster indices\n    @param X: np.ndarray\n        The dataset to cluster, where each row is a sample and each column is a feature.\n    @param k: int\n        The number of clusters the algorithm will form.\n    @param max_iterations: int\n        The number of iterations the algorithm will run for if it does\n        not converge before that.\n    "
    nsamples, features = X.shape
    centroids = X[np.random.choice(nsamples, k)]
    clusters = list[list[int]]()
    with persist.Loader(__file__, locals()) as transaction:
        if transaction:
            [
                X,
                _,
                centroids,
                cluster,
                cluster_i,
                clusters,
                diff,
                features,
                i,
                k,
                max_iterations,
                nsamples,
                prev_centroids,
                r,
                sample_i,
                y_pred,
            ] = transaction.move()
        for i in transaction.iterate(range(max_iterations)):  # type: int
            clusters = [list[int]() for _ in range(k)]
            for sample_i in range(len(X)):
                r = np.argmin(np.linalg.norm(X[sample_i] - centroids, axis=1))
                clusters[r].append(sample_i)
            prev_centroids = centroids
            centroids = np.array([np.mean(X[cluster], axis=0) for cluster in clusters])
            diff = centroids - prev_centroids
            if not diff.any():
                break
            transaction.commit(
                X,
                _,
                centroids,
                cluster,
                cluster_i,
                clusters,
                diff,
                features,
                i,
                k,
                max_iterations,
                nsamples,
                prev_centroids,
                r,
                sample_i,
                y_pred,
            )
    y_pred = np.zeros(nsamples)
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
