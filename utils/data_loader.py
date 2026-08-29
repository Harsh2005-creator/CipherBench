"""
Data loading utilities for CipherBench project.
Handles loading binary and multiclass cryptographic algorithm datasets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import yaml


def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_binary_dataset(algorithm_pair: str, size: str, test_size: float = 0.2,
                        random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load binary classification dataset for a specific algorithm pair and ciphertext size.

    Args:
        algorithm_pair: Algorithm pair name (e.g., "AES and 3DES")
        size: Ciphertext size folder (e.g., "1kb", "8kb", "64kb", "256kb", "512kb")
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    config = load_config()
    base_path = config['paths']['data_binary']

    # Construct file path
    file_path = os.path.join(base_path, size.lower(), f"{algorithm_pair}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values   # Last column (label)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def load_multiclass_dataset(size: str, test_size: float = 0.2,
                            random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load multiclass classification dataset for a specific ciphertext size.

    Args:
        size: Ciphertext size (e.g., "1KB", "8KB", "64KB", "256KB", "512KB")
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    config = load_config()
    base_path = config['paths']['data_multiclass']

    # Construct file path
    file_path = os.path.join(base_path, f"{size}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except last (10 NIST features)
    y = df.iloc[:, -1].values   # Last column (label: 0-4)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def load_hknnrf_split(size: str, test_size: float = 0.2,
                      random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray]:
    """
    Load multiclass dataset with additional split for HKNNRF training.
    Training data is split 50/50 for RF and KNN stages.

    Args:
        size: Ciphertext size (e.g., "1KB", "8KB", "64KB", "256KB", "512KB")
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducibility

    Returns:
        X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test
    """
    # First split into train and test
    X_train, X_test, y_train, y_test = load_multiclass_dataset(size, test_size, random_state)

    # Further split training data 50/50 for RF and KNN
    X_train_rf, X_train_knn, y_train_rf, y_train_knn = train_test_split(
        X_train, y_train, test_size=0.5, random_state=random_state
    )

    return X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test


def get_all_binary_pairs():
    """Get all available binary classification algorithm pairs."""
    config = load_config()
    algorithms = config['algorithms']['names']

    pairs = []
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            pairs.append(f"{algorithms[i]} and {algorithms[j]}")

    return pairs


def get_algorithm_name(label: int) -> str:
    """Convert numeric label to algorithm name."""
    config = load_config()
    return config['algorithms']['label_mapping'][label]


def get_feature_names():
    """Get NIST feature names."""
    return [
        'aetPValue', 'custPValue', 'dtfPValue', 'fwbtPValue',
        'lrobPValue', 'mtPValue', 'retPValue', 'revtPValue',
        'runsPValue', 'stPValue'
    ]


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")

    # Test multiclass loading
    print("\n1. Loading multiclass dataset (512KB)...")
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB")
    print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"   Unique labels: {np.unique(y_train)}")

    # Test binary loading
    print("\n2. Loading binary dataset (AES and 3DES, 1kb)...")
    X_train, X_test, y_train, y_test = load_binary_dataset("AES and 3DES", "1kb")
    print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"   Unique labels: {np.unique(y_train)}")

    # Test HKNNRF split
    print("\n3. Loading HKNNRF split (512KB)...")
    X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = load_hknnrf_split("512KB")
    print(f"   RF train: {X_train_rf.shape}, KNN train: {X_train_knn.shape}, Test: {X_test.shape}")

    # List all binary pairs
    print("\n4. All binary algorithm pairs:")
    for pair in get_all_binary_pairs():
        print(f"   - {pair}")

    print("\n✓ Data loader tests completed successfully!")
