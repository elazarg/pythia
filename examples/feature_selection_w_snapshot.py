import persist
import numpy as np

import type_system


def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, type_system.T) - y
    cost = 1 / (2 * m) * np.dot(type_system.T, error)
    return cost, error


def gradient_descent(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta = theta - (alpha * (1 / m) * np.dot(type_system.T, error))
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


def Linear_Regression(features: np.ndarray, target: np.ndarray, dims) -> np.ndarray:
    # preprocess features and target
    target = np.array(target).reshape(target.shape[0], -1)
    if features[:, dims].size > 0:
        # define sparse features
        sparse_features = features[:, dims]
        if sparse_features.ndim == 1:
            sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)
        predict = run(sparse_features, target)
    else:
        predict = np.zeros(features.shape[0]).reshape(features.shape[0], -1)
    grad = np.dot(features.T, target - predict)
    return grad


def do_work(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    # define rounds
    rounds = 0
    rounds_ind = 0

    # define new solution
    S = np.array([], int)
    persist._mark(S)

    for idx in range(k):
        if persist._now_recovering():
            idx, S = persist._recover(
                snapshot
            )  # load snapshot and set local variables to saved state
        persist._mark(idx)

        # define and train model
        # preprocess current solution
        grad = Linear_Regression(features, target, np.unique(S[S >= 0]))
        rounds += 1

        # define vals
        A = np.array(range(len(grad)))
        point = []
        for a in np.setdiff1d(A, S):
            point = np.append(point, a)
        out = [[point, len(np.setdiff1d(A, S))]]
        out = np.array(out, dtype="object")
        rounds_ind += np.max(out[:, -1])

        # get feasible points
        points = np.array([])
        points = np.append(points, np.array(out[0, 0]))
        points = points.astype("int")
        # break if points are no longer feasible
        if len(points) == 0:
            break

        # otherwise add maximum point to current solution
        a = points[0]
        for i in points:
            if grad[i] > grad[a]:
                a = i

        if grad[a] >= 0:
            persist._unmark(S)
            S = np.unique(np.append(S, a))
            persist._mark_shallow(
                S
            )  # for now S must be ndarray (can use pickle when committing)
        else:
            break

        persist._commit(idx, S)
    return S


def main():
    dataset_name = "dataset_20KB"
    features = np.load(dataset_name + "_features.npy")
    target = np.load(dataset_name + "_target.npy")
    do_work(features, target, 10)
