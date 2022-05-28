from __future__ import annotations

import sklearn as sk
import pymm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def score(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    error = np.dot(X, theta.T) - y
    return np.dot(error.T, error)


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


def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()


def predict(X, theta):
    predict_ = np.zeros((len(X), 1))
    for j in range(len(X)):
        x = X[j]
        sum_ = 0
        for i in range(len(x)):
            sum_ += x[i] * theta[i]
        predict_[j] = sum_
    return predict_


def linear_regression_mult_var_run(X, y):
    # Import data

    y = np.concatenate(y)
    # Normalize our features

    # Normalize
    X = (X - X.mean()) / X.std()
    # Add a 1 column to the start to allow vectorized gradient descent
    X = np.c_[np.ones(X.shape[0]), X]

    # Set hyperparameters
    alpha = 0.1
    iterations = 10000
    # Initialize Theta Values to 0
    theta = np.zeros(X.shape[1])
    initial_cost, _ = cost_function(X, y, theta)

    #    print('With initial theta values of {0}, cost error is {1}'.format(theta, initial_cost))

    # Run Gradient Descent
    #    theta1, cost_num = gradient_descent(X1, y1, theta1, alpha, iterations)
    theta, cost_num = gradient_descent(X, y, theta, alpha, iterations)

    # Display cost chart
    #    plotChart(iterations, cost_num)

    final_cost, _ = cost_function(X, y, theta)

    #    print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))
    return final_cost, predict(X, theta)


def oracle(features: np.ndarray, target: np.ndarray, S: np.ndarray, model: str):
    """
    Train the model and outputs metric for a set of features

    INPUTS:
    features -- the feature matrix
    target -- the observations
    S -- index for the features used for model construction
    model -- choose if the regression is linear or logistic
    OUTPUTS:
    float grad -- the garadient of the log-likelihood function
    float log_loss -- the log-loss for the trained model
    float -log_loss -- the negative log-loss, which is proportional to the log-likelihood
    float score -- the R^2 score for the trained linear model
    """
    # preprocess current solution
    S = np.unique(S[S >= 0])

    # logistic model
    if model == 'logistic':
        grad, log_like = Logistic_Regression(features, target, S)
        return grad, log_like

    # linear model
    else:
        assert model == 'linear'
        grad, score = Linear_Regression(features, target, S)
        #        print (score)
        return grad, score


def Logistic_Regression(features: np.ndarray, target: np.ndarray, dims: np.ndarray):
    """
    Logistic regression for a given set of features

    INPUTS:
    features -- the feature matrix
    target -- the observations
    dims -- index for the features used for model construction
    GRAD -- if set to TRUE the function returns grad
    float grad -- the gradient of the log-likelihood function
    float log_loss -- the log-loss for the trained model
    """

    if features[:, dims].size > 0:
        # define sparse features
        sparse_features = np.array(features[:, dims])
        if sparse_features.ndim == 1:
            sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)

        # get model, predict probabilities, and predictions
        model = sk.linear_model.LogisticRegression(max_iter=10000).fit(sparse_features, target)
        predict_prob = np.array(model.predict_proba(sparse_features))
        predictions = model.predict(sparse_features)

    else:
        # predict probabilities, and predictions
        predict_prob = np.ones((features.shape[0], 2)) * 0.5
        predictions = np.ones((features.shape[0])) * 0.5

    # conpute gradient of log likelihood

    log_like = len(target) * (
                 -sk.metrics.log_loss(target, predict_prob)
                 + sk.metrics.log_loss(target, np.ones((features.shape[0], 2)) * 0.5)
              )
    grad = np.dot(features.T, target - predictions)
    return grad, log_like


def Linear_Regression(features: np.ndarray, target: np.ndarray, dims: np.ndarray):
    '''
    Linear regression for a given set of features

    INPUTS:
    features -- the feature matrix
    target -- the observations
    dims -- index for the features used for model construction

    OUTPUTS:
    float grad -- the gradient of the log-likelihood function
    float score -- the R^2 score for the trained model
    '''

    target = np.array(target).reshape(target.shape[0], -1)
    if features[:, dims].size > 0:
        sparse_features = features[:, dims]
        if sparse_features.ndim == 1:
            sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)
        score, predict = linear_regression_mult_var_run(sparse_features, target)
    else:
        # predict probabilities, and predictions
        score = 0
        predict = np.zeros(features.shape[0]).reshape(features.shape[0], -1)
    # compute gradient of log likelihood
    grad = np.dot(features.T, target - predict)
    return grad, score


def do_work(features: np.ndarray, target: np.ndarray, model: str, k: int, recovery: int):
    s = global_shelf
    '''
    The SDS algorithm, as in "Submodular Dictionary Selection for Sparse Representation", Krause and Cevher, ICML '10

    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    '''

    # save data to file
    results = pd.DataFrame(
        data={'k': np.zeros(k).astype('int'), 'time': np.zeros(k), 'rounds_ind': np.zeros(k), 'rounds': np.zeros(k),
              'metric': np.zeros(k)})

    # define time and rounds
    rounds = 0
    rounds_ind = 0

    # initial checkpoint
    if recovery:
        S = s.S[k - s.idx[0]:k]
    else:
        s.S = np.zeros(k, dtype=int)
        s.idx = np.zeros(1, dtype=int)
        s.done = np.zeros(1, dtype=int)
        s.res_k = np.zeros(k, dtype=int)
        s.res_time = np.zeros(k)
        s.res_rounds = np.zeros(k)
        s.res_rounds_ind = np.zeros(k)
        s.res_metric = np.zeros(k)
        # define new solution
        S = np.array([], int)

    start_idx = s.idx[0]
    for idx in range(start_idx, k):

        # define and train model
        grad, metric = oracle(features, target, S, model)
        rounds += 1

        # define vals
        point = []
        A = np.array(range(len(grad)))
        for a in np.setdiff1d(A, S):
            point = np.append(point, a)
        out = [[point, len(np.setdiff1d(A, S))]]

        if idx == 2:
            print("exit where idx=", idx)
            print(s.S)

        out = np.array(out, dtype='object')
        rounds_ind += np.max(out[:, -1])
        # save results to file
        s.res_k[idx] = idx + 1
        s.res_time[idx] = results.loc[idx, 'time']
        s.res_rounds[idx] = int(rounds)
        s.res_rounds_ind[idx] = rounds_ind
        s.res_metric[idx] = metric

        # get feasible points
        points = np.array([])
        points = np.append(points, np.array(out[0, 0]))
        points = points.astype('int')
        # break if points are no longer feasible
        if not points:
            break

        # otherwise add maximum point to current solution
        a = points[0]
        for i in points:
            if grad[i] > grad[a]:
                a = i

        if grad[a] >= 0:
            S = np.unique(np.append(S, i))
        else:
            s.done[0] = 1
            break
        s.S[k - idx - 1] = S[0]
        s.idx[0] = s.idx + 1
        if s.idx == k and s.done == 0:
            s.done[0] = 1

    results['k'] = s.res_k
    results['time'] = s.res_time
    results['rounds'] = s.res_rounds
    results['rounds_ind'] = s.res_rounds_ind
    results['metric'] = s.res_metric
    return results


global_shelf = None


def main():
    global global_shelf
    """
    Test algorithms with the run_experiment function.
    target -- the observations for the regression
    features -- the feature matrix for the regression
    model -- choose if 'logistic' or 'linear' regression
    k_range -- range for the parameter k for a set of experiments
    """

    '''
    Linear Regression
    '''

    # define features and target for the experiments

    dataset_name = "dataset_20KB"

    # open shelf

    shelf_size_GB = 20
    shelf_path = '/mnt/pmem0'
    shelf_name = str(shelf_size_GB) + "GBshelf"  # 20GBshelf
    shelf_force_new = False
    # shelf_force_new = True
    global_shelf = pymm.shelf(shelf_name, size_mb=shelf_size_GB * 1024, pmem_path=shelf_path, force_new=shelf_force_new)

    ONLY_DRAM = 1
    print("Load target from")
    target_name = dataset_name + "_target.npy"

    print("Load target from: " + target_name)
    if ONLY_DRAM:
        target = np.load(target_name)  # DRAM only
    else:
        target = np.load(target_name, mmap_mode='r')  # DRAM + NVMe

    features_name = dataset_name + "_features.npy"

    if ONLY_DRAM:
        features = np.load(features_name)  # DRAM only
    else:
        features = np.load(features_name, mmap_mode='r')  # DRAM + NVMe

    # initalize features and target

    # choose if logistic or linear regression
    model = 'linear'

    # set range for the experiments
    k_range = np.array([10])

    # run experiment
    recovery = 0
    if global_shelf.idx is not None and global_shelf.idx > 0:
        if global_shelf.done is not None and (global_shelf.done or global_shelf.idx == k_range[-1]):  # done
            print("We finished this do_work")
            print("done =", global_shelf.done[0], ", idx =", global_shelf.idx[0])
            exit(0)
        else:
            print("start recovery process, idx={}", global_shelf.idx)
            recovery = 1
    print(recovery)
    do_work(features, target, model, k_range[-1], recovery)


if __name__ == '__main__':
    main()
