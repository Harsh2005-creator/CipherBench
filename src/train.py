"""
Main training script for CipherBench.
Train all models on binary and multiclass datasets.
"""

import argparse
import time
import numpy as np
from typing import List, Dict
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import (
    load_binary_dataset, load_multiclass_dataset,
    load_hknnrf_split, load_binary_hknnrf_split, get_all_binary_pairs, load_config
)
from src.utils.metrics import compute_metrics, compute_confusion_matrix
from src.models.baseline import get_svm_model, get_knn_model, get_random_forest_model
from src.models.hknnrf import HKNNRFClassifier
from src.models.mlp import MLPClassifier
from src.models.cnn import CNN1DClassifier
from database.db_operations import ExperimentDB


def train_baseline_model(model, model_name: str, X_train, y_train, X_test, y_test,
                         task_type: str, algorithm_pair: str, size: str, db: ExperimentDB):
    """Train and evaluate a baseline model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {algorithm_pair} ({size})")
    print(f"{'='*60}")

    start_time = time.time()

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    training_time = time.time() - start_time

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, task_type)
    cm = compute_confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Training time: {training_time:.2f}s")

    # Save to database
    try:
        db.insert_experiment(
            model_name=model_name,
            task_type=task_type,
            algorithm_pair=algorithm_pair,
            ciphertext_size=size,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            training_time=training_time,
            hyperparameters=model.get_params() if hasattr(model, 'get_params') else {},
            confusion_matrix=cm
        )
    except Exception as e:
        print(f"Warning: Could not save to database: {e}")

    return metrics


def train_hknnrf_model(X_train_rf, y_train_rf, X_train_knn, y_train_knn,
                       X_test, y_test, task_type: str, algorithm_pair: str,
                       size: str, db: ExperimentDB, config: Dict):
    """Train and evaluate HKNNRF model."""
    print(f"\n{'='*60}")
    print(f"Training HKNNRF on {algorithm_pair} ({size})")
    print(f"{'='*60}")

    start_time = time.time()

    # Get hyperparameters from config
    hparams = config['hyperparameters']['hknnrf']

    # Use mid-range values for default training
    model = HKNNRFClassifier(
        n_estimators=85,
        max_depth=5,
        n_neighbors=7
    )

    # Train
    model.fit(X_train_rf, y_train_rf, X_train_knn, y_train_knn)

    # Predict
    y_pred = model.predict(X_test)

    training_time = time.time() - start_time

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, task_type)
    cm = compute_confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Training time: {training_time:.2f}s")

    # Save to database
    try:
        db.insert_experiment(
            model_name="HKNNRF",
            task_type=task_type,
            algorithm_pair=algorithm_pair,
            ciphertext_size=size,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            training_time=training_time,
            hyperparameters=model.get_params(),
            confusion_matrix=cm
        )
    except Exception as e:
        print(f"Warning: Could not save to database: {e}")

    return metrics


def train_deep_model(ModelClass, model_name: str, X_train, y_train, X_test, y_test,
                     task_type: str, algorithm_pair: str, size: str,
                     db: ExperimentDB, config: Dict):
    """Train and evaluate a deep learning model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {algorithm_pair} ({size})")
    print(f"{'='*60}")

    start_time = time.time()

    # Determine number of classes
    num_classes = 2 if task_type == 'binary' else 5

    # Get hyperparameters
    hparams_key = 'mlp' if model_name == 'MLP' else 'cnn'
    hparams = config['hyperparameters'][hparams_key]

    # Initialize model
    model = ModelClass(
        num_classes=num_classes,
        epochs=hparams['epochs'],
        batch_size=hparams['batch_size'],
        learning_rate=hparams['learning_rate'],
        dropout_rate=hparams['dropout_rate'],
        verbose=1
    )

    # Train
    model.fit(X_train, y_train, X_test, y_test)

    # Predict
    y_pred = model.predict(X_test)

    training_time = time.time() - start_time

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, task_type)
    cm = compute_confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Training time: {training_time:.2f}s")

    # Save to database
    try:
        db.insert_experiment(
            model_name=model_name,
            task_type=task_type,
            algorithm_pair=algorithm_pair,
            ciphertext_size=size,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            training_time=training_time,
            hyperparameters=model.get_params(),
            confusion_matrix=cm
        )
    except Exception as e:
        print(f"Warning: Could not save to database: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train CipherBench models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'svm', 'knn', 'rf', 'hknnrf', 'mlp', 'cnn'],
                       help='Model to train')
    parser.add_argument('--task', type=str, default='multiclass',
                       choices=['all', 'binary', 'multiclass'],
                       help='Task type')
    parser.add_argument('--size', type=str, default='512KB',
                       choices=['all', '1KB', '8KB', '64KB', '256KB', '512KB'],
                       help='Ciphertext size')

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Initialize database
    db = ExperimentDB()

    # Determine sizes to train
    sizes = config['ciphertext_sizes'] if args.size == 'all' else [args.size]

    # Determine tasks
    tasks = ['binary', 'multiclass'] if args.task == 'all' else [args.task]

    # Train models
    for size in sizes:
        for task in tasks:
            if task == 'multiclass':
                # Load multiclass data
                X_train, X_test, y_train, y_test = load_multiclass_dataset(size)

                # Train baseline models
                if args.model in ['all', 'svm']:
                    model = get_svm_model(**config['hyperparameters']['svm'])
                    train_baseline_model(model, 'SVM', X_train, y_train, X_test, y_test,
                                        'multiclass', '5-class', size, db)

                if args.model in ['all', 'knn']:
                    model = get_knn_model(**config['hyperparameters']['knn'])
                    train_baseline_model(model, 'KNN', X_train, y_train, X_test, y_test,
                                        'multiclass', '5-class', size, db)

                if args.model in ['all', 'rf']:
                    model = get_random_forest_model(**config['hyperparameters']['random_forest'])
                    train_baseline_model(model, 'RF', X_train, y_train, X_test, y_test,
                                        'multiclass', '5-class', size, db)

                # Train HKNNRF
                if args.model in ['all', 'hknnrf']:
                    X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = load_hknnrf_split(size)
                    train_hknnrf_model(X_train_rf, y_train_rf, X_train_knn, y_train_knn,
                                      X_test, y_test, 'multiclass', '5-class', size, db, config)

                # Train deep learning models
                if args.model in ['all', 'mlp']:
                    train_deep_model(MLPClassifier, 'MLP', X_train, y_train, X_test, y_test,
                                    'multiclass', '5-class', size, db, config)

                if args.model in ['all', 'cnn']:
                    train_deep_model(CNN1DClassifier, 'CNN', X_train, y_train, X_test, y_test,
                                    'multiclass', '5-class', size, db, config)

            elif task == 'binary':
                # For binary, train on all pairs (paper tests all 10 combinations)
                all_pairs = get_all_binary_pairs()

                for pair in all_pairs:
                    try:
                        # Convert size format (1KB -> 1kb for binary folder names)
                        size_folder = size.lower().replace('kb', 'kb')

                        if args.model in ['all', 'svm', 'knn', 'rf']:
                            # Load standard binary data for baseline models
                            X_train, X_test, y_train, y_test = load_binary_dataset(pair, size_folder)

                            if args.model in ['all', 'svm']:
                                model = get_svm_model(**config['hyperparameters']['svm'])
                                train_baseline_model(model, 'SVM', X_train, y_train, X_test, y_test,
                                                    'binary', pair, size, db)

                            if args.model in ['all', 'knn']:
                                model = get_knn_model(**config['hyperparameters']['knn'])
                                train_baseline_model(model, 'KNN', X_train, y_train, X_test, y_test,
                                                    'binary', pair, size, db)

                            if args.model in ['all', 'rf']:
                                model = get_random_forest_model(**config['hyperparameters']['random_forest'])
                                train_baseline_model(model, 'RF', X_train, y_train, X_test, y_test,
                                                    'binary', pair, size, db)

                        # HKNNRF for binary classification
                        if args.model in ['all', 'hknnrf']:
                            X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = load_binary_hknnrf_split(pair, size_folder)
                            train_hknnrf_model(X_train_rf, y_train_rf, X_train_knn, y_train_knn,
                                              X_test, y_test, 'binary', pair, size, db, config)

                    except FileNotFoundError:
                        print(f"Skipping {pair} ({size}) - file not found")
                        continue

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
