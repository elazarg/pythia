from __future__ import annotations
import math as mt
import numpy as np
import pandas as pd
import time as time
import sklearn.metrics as mt
from sklearn.linear_model import *
import linear_regression_mult_var
import pymm

is_sklearn = 0


def oracle(features: np.ndarray, target: np.ndarray, S: np.ndarray, model: str):
    '''
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
    '''
    # preprocess current solution
    S = np.unique(S[S >= 0])

    # logistic model
    if model == 'logistic':
        grad, log_like = Logistic_Regression(features, target, S)
        return grad, log_like

    # linear model
    if model == 'linear':
        grad, score = Linear_Regression(features, target, S)
        #        print (score)
        return grad, score


# ------------------------------------------------------------------------------------------
#  logistic regression
# ------------------------------------------------------------------------------------------

def Logistic_Regression(features: np.ndarray, target: np.ndarray, dims: int):
    '''
    Logistic regression for a given set of features

    INPUTS:
    features -- the feature matrix
    target -- the observations
    dims -- index for the features used for model construction
    GRAD -- if set to TRUE the function returns grad
    float grad -- the garadient of the log-likelihood function
    float log_loss -- the log-loss for the trained model
    '''

    if (features[:, dims].size > 0):

        # define sparse features
        sparse_features = np.array(features[:, dims])
        if sparse_features.ndim == 1: sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)

        # get model, predict probabilities, and predictions
        model = LogisticRegression(max_iter=10000).fit(sparse_features, target)
        predict_prob = np.array(model.predict_proba(sparse_features))
        predictions = model.predict(sparse_features)

    else:

        # predict probabilities, and predictions
        predict_prob = np.ones((features.shape[0], 2)) * 0.5
        predictions = np.ones((features.shape[0])) * 0.5

    # conpute gradient of log likelihood

    log_like = (-mt.log_loss(target, predict_prob) + mt.log_loss(target, np.ones((features.shape[0], 2)) * 0.5)) * len(
        target)
    grad = np.dot(features.T, target - predictions)
    return grad, log_like


# ------------------------------------------------------------------------------------------
#  linear regression
# ------------------------------------------------------------------------------------------

def Linear_Regression(features: np.ndarray, target: np.ndarray, dims: int):
    '''
    Linear regression for a given set of features

    INPUTS:
    features -- the feature matrix
    target -- the observations
    dims -- index for the features used for model construction

    OUTPUTS:
    float grad -- the garadient of the log-likelihood function
    float score -- the R^2 score for the trained model
    '''

    target = np.array(target).reshape(target.shape[0], -1)
    if (features[:, dims].size > 0):

        sparse_features = features[:, dims]
        if sparse_features.ndim == 1: sparse_features = sparse_features.reshape(sparse_features.shape[0], 1)
        if (is_sklearn):
            model = LinearRegression().fit(sparse_features, target)
            score = model.score(sparse_features, target)
            predict = model.predict(sparse_features)
        else:
            score, predict = linear_regression_mult_var.run(sparse_features, target)
    else:
        # predict probabilities, and predictions
        score = 0
        predict = (np.zeros((features.shape[0]))).reshape(features.shape[0], -1)
    # compute gradient of log likelihood
    grad = np.dot(features.T, target - predict)
    return grad, score


def do_work(featuers: np.ndarray, target: np.ndarray, model: str, k: int, recovery: int):
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
    run_time = time.time()
    rounds = 0
    rounds_ind = 0

    # intial checkpoint
    if (not recovery):
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

    else:  # recovery
        print(s.S[0:4])
        S = s.S[k - s.idx[0]:k]
        print(S)
        print(s.S)

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

        if (idx == 2):
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
        e = time.time()
        # break if points are no longer feasible
        if len(points) == 0: break

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
        print(idx)
        print(S)
        print(s.S)
        s.S[k - idx - 1] = S[0]
        s.idx[0] = s.idx + 1
        if (s.idx == k and s.done == 0):
            s.done[0] = 1

    results['k'] = s.res_k
    results['time'] = s.res_time
    results['rounds'] = s.res_rounds
    results['rounds_ind'] = s.res_rounds_ind
    results['metric'] = s.res_metric

    print(results)
    return results


'''
Test algorithms with the run_experiment function.
target -- the observations for the regression
features -- the feature matrix for the regression
model -- choose if 'logistic' or 'linear' regression
k_range -- range for the parameter k for a set of experiments
'''

'''
Linear Regration
'''

# define features and target for the experiments

dataset_name = "dataset_20KB"

# open shelf

shelf_size_GB = 20
shelf_path = '/mnt/pmem0'
shelf_name = str(shelf_size_GB) + "GBshelf"  # 20GBshelf
shelf_force_new = False
# shelf_force_new = True
s = pymm.shelf(shelf_name, size_mb=shelf_size_GB * 1024, pmem_path=shelf_path, force_new=shelf_force_new)

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
if (s.idx is not None and s.idx > 0):
    if (s.done is not None and (s.done or s.idx == k_range[-1])):  # done
        print("We finished this do_work")
        print("done =", s.done[0], ", idx =", s.idx[0])
        exit(0)
    else:
        print("start recovery process, idx={}", s.idx)
        recovery = 1
print(recovery)
do_work(features, target, model, k_range[-1], recovery)