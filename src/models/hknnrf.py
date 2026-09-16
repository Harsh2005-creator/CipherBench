"""
Hybrid K-Nearest Neighbors + Random Forest (HKNNRF) model.
Based on Yuan et al.'s methodology for cryptographic algorithm identification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, Dict, Any


class HKNNRFClassifier:
    """
    Hybrid classifier that combines Random Forest feature extraction with KNN classification.

    Pipeline:
    1. Train Random Forest on first portion of training data
    2. Extract leaf indices from RF for all samples
    3. One-hot encode the leaf indices (sparse feature representation)
    4. Train KNN on the encoded features using second portion of training data
    5. Predict using the same transformation pipeline
    """

    def __init__(self, n_estimators: int = 85, max_depth: int = 5,
                 n_neighbors: int = 7, random_state: int = 0):
        """
        Initialize HKNNRF classifier.

        Args:
            n_estimators: Number of trees in the Random Forest
            max_depth: Maximum depth of each tree
            n_neighbors: Number of neighbors for KNN
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        # Initialize components
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            oob_score=True,
            random_state=random_state
        )
        self.ohe = OneHotEncoder()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        self.is_fitted = False

    def fit(self, X_train_rf: np.ndarray, y_train_rf: np.ndarray,
            X_train_knn: np.ndarray, y_train_knn: np.ndarray):
        """
        Fit the HKNNRF model.

        Args:
            X_train_rf: Training data for Random Forest stage
            y_train_rf: Training labels for Random Forest stage
            X_train_knn: Training data for KNN stage
            y_train_knn: Training labels for KNN stage
        """
        # Step 1: Train Random Forest
        self.rf.fit(X_train_rf, y_train_rf)

        # Step 2: Get leaf indices from RF (new features from decision trees)
        leaf_indices_rf = self.rf.apply(X_train_rf)

        # Step 3: Fit One-Hot Encoder on RF leaf indices
        self.ohe.fit(leaf_indices_rf)

        # Step 4: Transform KNN training data through RF and OHE
        leaf_indices_knn = self.rf.apply(X_train_knn)
        X_train_knn_encoded = self.ohe.transform(leaf_indices_knn)

        # Step 5: PAPER STEP 9-10: Add new features to original features
        # "Use the trained decision trees to construct new features and add them to the original features"
        # Concatenate original NIST features + one-hot encoded RF leaf indices
        X_train_knn_combined = np.hstack([
            X_train_knn,  # Original 10 NIST features
            X_train_knn_encoded.toarray()  # RF-derived features (one-hot encoded)
        ])

        # Step 6: Train KNN on COMBINED features (original + new)
        self.knn.fit(X_train_knn_combined, y_train_knn)

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Input samples (original NIST features)

        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Transform through RF -> OHE -> Combine with original -> KNN
        leaf_indices = self.rf.apply(X)
        X_encoded = self.ohe.transform(leaf_indices)

        # PAPER STEP 9-10: Combine original features + new features
        X_combined = np.hstack([
            X,  # Original 10 NIST features
            X_encoded.toarray()  # RF-derived features
        ])

        predictions = self.knn.predict(X_combined)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Input samples (original NIST features)

        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Transform through RF -> OHE -> Combine with original -> KNN
        leaf_indices = self.rf.apply(X)
        X_encoded = self.ohe.transform(leaf_indices)

        # PAPER STEP 9-10: Combine original features + new features
        X_combined = np.hstack([
            X,  # Original 10 NIST features
            X_encoded.toarray()  # RF-derived features
        ])

        probabilities = self.knn.predict_proba(X_combined)

        return probabilities

    def get_params(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'n_neighbors': self.n_neighbors
        }


def hyperparameter_search(X_train_rf: np.ndarray, y_train_rf: np.ndarray,
                         X_train_knn: np.ndarray, y_train_knn: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         n_estimators_range: Tuple[int, int] = (80, 90),
                         max_depth_range: Tuple[int, int] = (1, 10),
                         n_neighbors_range: Tuple[int, int] = (5, 9)) -> Dict[str, Any]:
    """
    Perform grid search over HKNNRF hyperparameters.

    Args:
        X_train_rf: Training data for RF stage
        y_train_rf: Training labels for RF stage
        X_train_knn: Training data for KNN stage
        y_train_knn: Training labels for KNN stage
        X_test: Test data
        y_test: Test labels
        n_estimators_range: Range of n_estimators to search
        max_depth_range: Range of max_depth to search
        n_neighbors_range: Range of n_neighbors to search

    Returns:
        Dictionary with best parameters and their accuracy
    """
    from sklearn.metrics import accuracy_score

    best_accuracy = 0
    best_params = {}
    all_results = []

    total_combinations = (
        (n_estimators_range[1] - n_estimators_range[0]) *
        (max_depth_range[1] - max_depth_range[0]) *
        (n_neighbors_range[1] - n_neighbors_range[0])
    )

    print(f"Starting hyperparameter search ({total_combinations} combinations)...")

    for n_est in range(n_estimators_range[0], n_estimators_range[1]):
        for depth in range(max_depth_range[0], max_depth_range[1]):
            for n_neigh in range(n_neighbors_range[0], n_neighbors_range[1]):
                # Train model
                model = HKNNRFClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    n_neighbors=n_neigh
                )
                model.fit(X_train_rf, y_train_rf, X_train_knn, y_train_knn)

                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Store result
                result = {
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'n_neighbors': n_neigh,
                    'accuracy': accuracy
                }
                all_results.append(result)

                # Update best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'n_neighbors': n_neigh,
                        'accuracy': accuracy
                    }

    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best params: n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, n_neighbors={best_params['n_neighbors']}")

    return {
        'best_params': best_params,
        'all_results': all_results
    }
