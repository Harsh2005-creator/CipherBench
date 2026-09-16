"""
Baseline machine learning models wrapper for CipherBench.
Implements SVM, KNN, and Random Forest.
"""

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any


def get_svm_model(kernel: str = 'linear', gamma: float = 0.001) -> SVC:
    """
    Create and return Support Vector Machine classifier.

    Args:
        kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        gamma: Kernel coefficient
    """
    # probability=True allows predict_proba calls, useful for ROC/API
    return SVC(kernel=kernel, gamma=gamma, probability=True, random_state=0)


def get_knn_model(n_neighbors: int = 3) -> KNeighborsClassifier:
    """
    Create and return K-Nearest Neighbors classifier.

    Args:
         n_neighbors: Number of neighbors to use
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def get_random_forest_model(n_estimators: int = 20, max_depth: Any = None) -> RandomForestClassifier:
    """
    Create and return Random Forest classifier.

    Args:
        n_estimators: The number of trees in the forest
        max_depth: The maximum depth of the tree
    """
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)


def get_baseline_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get all baseline models configured.
    """
    hparams = config.get('hyperparameters', {})

    svm_h = hparams.get('svm', {'kernel': 'linear', 'gamma': 0.001})
    knn_h = hparams.get('knn', {'n_neighbors': 3})
    rf_h = hparams.get('random_forest', {'n_estimators': 20})

    return {
        'SVM': get_svm_model(kernel=svm_h['kernel'], gamma=svm_h['gamma']),
        'KNN': get_knn_model(n_neighbors=knn_h['n_neighbors']),
        'RF': get_random_forest_model(n_estimators=rf_h['n_estimators'])
    }
