"""
Visualization utilities for CipherBench.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


def plot_confusion_matrix(cm: np.ndarray, classes: List[str],
                         title: str = 'Confusion Matrix',
                         cmap=plt.cm.Blues,
                         save_path: str = None):
    """
    Plot confusion matrix with matplotlib.

    Args:
        cm: Confusion matrix
        classes: Class labels
        title: Plot title
        cmap: Color map
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_model_comparison(model_stats: Dict[str, Dict[str, float]],
                         metric: str = 'accuracy',
                         save_path: str = None):
    """
    Plot bar chart comparing models on a specific metric.

    Args:
        model_stats: Dictionary with model names as keys and metrics as values
        metric: Metric to plot
        save_path: Path to save figure (optional)
    """
    models = list(model_stats.keys())
    values = [stats[metric] for stats in model_stats.values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(models, values, color='steelblue')

    # Color the best model
    best_idx = values.index(max(values))
    bars[best_idx].set_color('gold')

    ax.set_xlabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'Model Comparison: {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history for deep learning models.

    Args:
        history: Training history dictionary
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    ax1.plot(history['accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {save_path}")

    return fig
