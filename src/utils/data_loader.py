"""
Data loading utilities for CipherBench.

The raw dataset labels are preserved. Binary labels may be 1/2 in the
source CSVs; individual models that require 0/1 (e.g. sigmoid networks)
perform their own internal encoding and map predictions back to the
original labels.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
import yaml


def load_config():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config.yaml",
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _validate_dataset(df: pd.DataFrame, expected_features: int = 10):
    if df.shape[1] != expected_features + 1:
        raise ValueError(
            f"Expected {expected_features} feature columns + 1 label column, "
            f"found {df.shape[1]} columns."
        )
    if df.iloc[:, :-1].isnull().any().any():
        raise ValueError("Dataset contains NaN feature values.")
    if not np.isfinite(df.iloc[:, :-1].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite feature values.")


def load_binary_dataset(
    algorithm_pair: str,
    size: str,
    test_size: float = 0.2,
    random_state: int = 0,
):
    config = load_config()
    base_path = config["paths"]["data_binary"]
    file_path = os.path.join(base_path, size.lower(), f"{algorithm_pair}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    _validate_dataset(df)

    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy()

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"Binary dataset {algorithm_pair} ({size}) has {len(classes)} "
            f"classes instead of 2: {classes}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def load_multiclass_dataset(
    size: str,
    test_size: float = 0.2,
    random_state: int = 0,
):
    config = load_config()
    base_path = config["paths"]["data_multiclass"]
    file_path = os.path.join(base_path, f"{size}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    _validate_dataset(df)

    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy()

    classes = np.unique(y)
    if len(classes) != 5:
        raise ValueError(
            f"Multiclass dataset {size} has {len(classes)} classes instead of 5: {classes}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def load_hknnrf_split(
    size: str,
    test_size: float = 0.2,
    random_state: int = 0,
):
    X_train, X_test, y_train, y_test = load_multiclass_dataset(
        size, test_size, random_state
    )
    X_train_rf, X_train_knn, y_train_rf, y_train_knn = train_test_split(
        X_train,
        y_train,
        test_size=0.5,
        random_state=random_state,
        stratify=y_train,
    )
    return X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test


def get_all_binary_pairs():
    config = load_config()
    algorithms = config["algorithms"]["names"]
    return [
        f"{algorithms[i]} and {algorithms[j]}"
        for i in range(len(algorithms))
        for j in range(i + 1, len(algorithms))
    ]


def get_algorithm_name(label: int) -> str:
    config = load_config()
    return config["algorithms"]["label_mapping"][label]


def get_feature_names():
    return [
        "aetPValue", "custPValue", "dtfPValue", "fwbtPValue",
        "lrobPValue", "mtPValue", "retPValue", "revtPValue",
        "runsPValue", "stPValue",
    ]


def load_binary_hknnrf_split(
    algorithm_pair: str,
    size: str,
    test_size: float = 0.2,
    random_state: int = 0,
):
    X_train, X_test, y_train, y_test = load_binary_dataset(
        algorithm_pair, size, test_size, random_state
    )
    X_train_rf, X_train_knn, y_train_rf, y_train_knn = train_test_split(
        X_train,
        y_train,
        test_size=0.5,
        random_state=random_state,
        stratify=y_train,
    )
    return X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test
