"""
Evaluation metrics computation utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any
import json


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   task_type: str = 'binary') -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: 'binary' or 'multiclass'

    Returns:
        Dictionary containing accuracy, precision, recall, F1-score
    """
    # For binary classification, use 'weighted' to handle any label values
    # This works for [0,1], [0,2], or any two-class scenario
    average = 'weighted' if task_type == 'binary' else 'weighted'

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def format_metrics_for_db(metrics: Dict[str, float], cm: np.ndarray) -> Dict[str, Any]:
    """
    Format metrics for database storage.

    Args:
        metrics: Dictionary of metric values
        cm: Confusion matrix

    Returns:
        Dictionary with serializable values
    """
    return {
        'accuracy': round(metrics['accuracy'], 4),
        'precision': round(metrics['precision'], 4),
        'recall': round(metrics['recall'], 4),
        'f1_score': round(metrics['f1_score'], 4),
        'confusion_matrix': json.dumps(cm.tolist())
    }


def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str, task_type: str = 'binary'):
    """
    Print comprehensive evaluation report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        task_type: 'binary' or 'multiclass'
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Report: {model_name}")
    print(f"{'='*60}")

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, task_type)

    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test metrics computation
    print("Testing metrics module...")

    # Binary classification example
    y_true_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_binary = np.array([0, 1, 0, 0, 0, 1, 1, 1])

    print("\n1. Binary Classification Metrics:")
    metrics = compute_metrics(y_true_binary, y_pred_binary, 'binary')
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")

    # Multiclass example
    y_true_multi = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred_multi = np.array([0, 1, 2, 3, 3, 0, 2, 2, 3, 4])

    print("\n2. Multiclass Classification Metrics:")
    metrics = compute_metrics(y_true_multi, y_pred_multi, 'multiclass')
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")

    print("\n3. Confusion Matrix:")
    cm = compute_confusion_matrix(y_true_multi, y_pred_multi)
    print(cm)

    print("\n[OK] Metrics module tests completed successfully!")
