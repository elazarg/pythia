import sys

import numpy as np
import argparse


def append_int(a: np.ndarray, n: int) -> np.ndarray:
    return np.append(a, n)


def get_float(array: np.ndarray, idx: int) -> float:
    res = array[idx]
    # assert isinstance(res, float)
    return res


def log(idx: int, k: int) -> None:
    print(f"{idx} / {k}", end="\r", flush=True, file=sys.stderr)


def run(features: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """Select k features from features using target as the target variable with predictable memory usage"""
    n_samples, n_features = features.shape

    # Pre-allocate arrays with known maximum sizes
    S = np.full(k, -1, dtype=int)  # Selected features, -1 indicates empty
    selected_count = 0

    # Pre-allocate working arrays to avoid repeated allocation
    prediction = np.zeros((n_samples, 1))
    available_mask = np.ones(n_features, dtype=bool)

    # Pre-allocate for linear regression
    max_X_cols = min(k, n_features) + 1  # +1 for bias term
    X_allocated = np.zeros((n_samples, max_X_cols))
    theta = np.zeros(max_X_cols)

    for idx in range(k):
        log(idx, k)

        # Prepare feature matrix using selected features
        if selected_count == 0:
            # No features selected yet, use zero predictions
            prediction.fill(0.0)
        else:
            # Extract selected features into pre-allocated array
            selected_features = S[:selected_count]
            X_data = features[:, selected_features]

            # Standardize features
            if X_data.ndim == 1:
                X_data = X_data.reshape(-1, 1)

            X_mean = np.mean(X_data, axis=0)
            X_std = np.std(X_data, axis=0)

            # Handle zero standard deviation
            X_std = np.where(X_std == 0, 1, X_std)

            X_standardized = (X_data - X_mean) / X_std

            # Add bias term to pre-allocated array
            n_cols = X_standardized.shape[1] + 1
            X_allocated[:, 0] = 1.0  # bias term
            X_allocated[:, 1:n_cols] = X_standardized

            # Use view of the allocated array
            X = X_allocated[:, :n_cols]

            # Flatten target for training
            y = target.flatten()

            # Reset theta for current number of features
            theta[:n_cols] = 0.0
            theta_view = theta[:n_cols]

            # Gradient descent
            learning_rate = 0.1 / len(y)
            for _ in range(10000):
                predictions = np.dot(X, theta_view)
                error = predictions - y
                gradient = np.dot(X.T, error)
                theta_view -= learning_rate * gradient

            # Make predictions using manual dot product (matching original logic)
            prediction.fill(0.0)
            for j in range(n_samples):
                total = 0.0
                for i in range(n_cols):
                    total += X[j, i] * theta_view[i]
                prediction[j, 0] = total

        # Calculate gradient for feature selection
        residual = target - prediction
        grad = np.dot(features.T, residual).flatten()

        # Update available features mask
        if selected_count > 0:
            available_mask[S[:selected_count]] = False

        # Find available feature indices
        available_indices = np.where(available_mask)[0]

        # Break if no more features available
        if len(available_indices) == 0:
            break

        # Find best feature among available ones
        available_grad_values = grad[available_indices]
        best_relative_idx = np.argmax(available_grad_values)
        best_feature_idx = available_indices[best_relative_idx]
        best_grad_value = available_grad_values[best_relative_idx]

        # Add feature if gradient is positive
        if best_grad_value >= 0:
            S[selected_count] = best_feature_idx
            available_mask[best_feature_idx] = False
            selected_count += 1
        else:
            break

    # Return only the selected features (remove unused slots)
    return S[:selected_count]


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
