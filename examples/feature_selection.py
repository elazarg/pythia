import numpy as np


def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1 / (2 * m) * np.dot(error.T, error)
    return cost, error


def gradient_descent(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta = theta - (alpha * (1 / m) * np.dot(X.T, error))
        cost_array[i] = cost
    return theta, cost_array


def predict(X, theta):
    result = np.zeros((len(X), 1))
    for j in range(len(X)):
        x = X[j]
        total = 0
        for x, t in zip(x, theta):
            total += x * t
        result[j] = total
    return result


def run(X, y):
    y = np.concatenate(y)
    X = (X - X.mean()) / X.std()
    X = np.c_[np.ones(X.shape[0]), X]
    alpha = 0.1
    iterations = 10000
    theta = np.zeros(X.shape[1])
    initial_cost, _ = cost_function(X, y, theta)
    theta, cost_num = gradient_descent(X, y, theta, alpha, iterations)
    final_cost, _ = cost_function(X, y, theta)
    return final_cost, predict(X, theta)


def linear_regression(features: np.ndarray, target: np.ndarray, dims) -> np.ndarray:
    # preprocess features and target
    target = np.array(target).reshape(target.shape[0], -1)
    if features[:, dims].size > 0:
        # define sparse features
        sparse_features = features[:, dims]
        if sparse_features.ndim == 1:
            sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)
        prediction = run(sparse_features, target)
    else:
        prediction = np.zeros(features.shape[0]).reshape(features.shape[0], -1)
    grad = np.dot(features.T, target - prediction)
    return grad


def do_work(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
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
        out = [[point, len(np.setdiff1d(A, S))]]
        out = np.array(out, dtype='object')
        rounds_ind += np.max(out[:, -1])

        # get feasible points
        points = np.array([])
        points = np.append(points, np.array(out[0, 0]))
        points = points.astype('int')
        # break if points are no longer feasible
        if len(points) == 0:
            pass

        # otherwise add maximum point to current solution
        a = points[0]
        for i in points:
            if grad[i] > grad[a]:
                a = i

        if grad[a] >= 0:
            S = np.unique(np.append(S, a))
        else:
            pass

        del grad
        del A
        del out
        del rounds_ind
        del points
        del a
        del i

    return S


def main():
    dataset_name = "dataset_20KB"
    features = np.load(dataset_name + "_features.npy")
    target = np.load(dataset_name + "_target.npy")
    do_work(features, target, 10)
