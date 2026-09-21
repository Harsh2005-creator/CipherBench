"""
Complete Experiment Runner for CipherBench
Executes the full experiment matrix with proper splitting and evaluation.
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
import argparse
from sklearn.model_selection import train_test_split

from src.utils.data_loader import (
    load_binary_dataset, load_multiclass_dataset,
    load_hknnrf_split, load_binary_hknnrf_split,
    get_all_binary_pairs, load_config
)
from src.utils.metrics import compute_metrics, compute_confusion_matrix
from src.models.baseline import get_svm_model, get_knn_model, get_random_forest_model
from src.models.svmnb import get_svmnb_model
from src.models.hknnrf import HKNNRFClassifier
from src.models.mlp import MLPClassifier
from src.models.cnn import CNN1DClassifier
from database.db_operations import ExperimentDB


class ExperimentRunner:
    """
    Manages complete experiment execution with reproducibility and result tracking.
    """

    def __init__(self, output_dir=None, use_db=True, random_seed=42):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory to save results
            use_db: Whether to save results to database
            random_seed: Random seed for reproducibility
        """
        self.output_dir = output_dir if output_dir is not None else DEFAULT_RESULTS_DIR
        self.use_db = use_db
        self.random_seed = random_seed
        self.config = load_config()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize database if needed
        if use_db:
            try:
                self.db = ExperimentDB()
            except Exception as e:
                print(f"Warning: Database connection failed: {e}")
                print("Results will only be saved to files.")
                self.use_db = False

        # Result storage
        self.results = []

    def _fit_deep_model(self, model, X_train, y_train, seed: int):
        """
        Fit an MLP/CNN model with a validation split carved out of the
        TRAINING data only. The caller's X_test/y_test must never be passed
        here: early stopping and model selection must not see the test set.
        """
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=seed,
            stratify=y_train,
        )
        model.fit(X_fit, y_fit, X_val, y_val)
        return model

    def run_binary_experiment(self, pair: str, size: str, model_name: str, seed: int = None) -> Dict:
        """
        Run single binary classification experiment.

        Args:
            pair: Algorithm pair (e.g., "AES and 3DES")
            size: Ciphertext size folder (e.g., "512kb")
            model_name: Model to train
            seed: Random seed (uses instance default if None)

        Returns:
            Dictionary with experiment results
        """
        seed = seed if seed is not None else self.random_seed

        print(f"\n{'='*70}")
        print(f"Binary: {pair} | Size: {size} | Model: {model_name} | Seed: {seed}")
        print(f"{'='*70}")

        try:
            start_time = time.time()

            # Load data based on model
            if model_name == 'HKNNRF':
                X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = \
                    load_binary_hknnrf_split(pair, size, test_size=0.2, random_state=seed)

                # Train HKNNRF
                model = HKNNRFClassifier(
                    n_estimators=self.config['hyperparameters']['hknnrf']['n_estimators_range'][0],
                    max_depth=self.config['hyperparameters']['hknnrf']['max_depth_range'][0],
                    n_neighbors=self.config['hyperparameters']['hknnrf']['n_neighbors_range'][0],
                    random_state=seed
                )
                model.fit(X_train_rf, y_train_rf, X_train_knn, y_train_knn)
                y_pred = model.predict(X_test)

            else:
                # Standard train/test split
                X_train, X_test, y_train, y_test = load_binary_dataset(
                    pair, size, test_size=0.2, random_state=seed
                )

                # Train model
                if model_name == 'SVM':
                    model = get_svm_model(**self.config['hyperparameters']['svm'])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif model_name == 'KNN':
                    model = get_knn_model(**self.config['hyperparameters']['knn'])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif model_name == 'RF':
                    model = get_random_forest_model(**self.config['hyperparameters']['random_forest'])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif model_name == 'SVMNB':
                    model = get_svmnb_model(**self.config['hyperparameters']['svmnb'], random_state=seed)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif model_name == 'MLP':
                    model = MLPClassifier(
                        num_classes=2,
                        **self.config['hyperparameters']['mlp'],
                        verbose=0,
                        random_state=seed,
                    )
                    self._fit_deep_model(model, X_train, y_train, seed)
                    y_pred = model.predict(X_test)

                elif model_name == 'CNN':
                    model = CNN1DClassifier(
                        num_classes=2,
                        **self.config['hyperparameters']['cnn'],
                        verbose=0,
                        random_state=seed,
                    )
                    self._fit_deep_model(model, X_train, y_train, seed)
                    y_pred = model.predict(X_test)

                else:
                    raise ValueError(f"Unknown model: {model_name}")

            training_time = time.time() - start_time

            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, 'binary')
            cm = compute_confusion_matrix(y_test, y_pred)

            # Package results
            result = {
                'model_name': model_name,
                'task_type': 'binary',
                'algorithm_pair': pair,
                'ciphertext_size': size,
                'train_size': len(y_train) if model_name != 'HKNNRF' else len(y_train_rf) + len(y_train_knn),
                'test_size': len(y_test),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'confusion_matrix': cm.tolist(),
                'training_time': training_time,
                'random_seed': seed,
                'timestamp': datetime.now().isoformat()
            }

            print(f"[OK] Accuracy: {metrics['accuracy']:.4f} | Time: {training_time:.2f}s")

            return result

        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_multiclass_experiment(self, size: str, model_name: str, seed: int = None) -> Dict:
        """
        Run single multiclass classification experiment.

        Args:
            size: Ciphertext size (e.g., "512KB")
            model_name: Model to train
            seed: Random seed

        Returns:
            Dictionary with experiment results
        """
        seed = seed if seed is not None else self.random_seed

        print(f"\n{'='*70}")
        print(f"Multiclass: 5-class | Size: {size} | Model: {model_name} | Seed: {seed}")
        print(f"{'='*70}")

        try:
            start_time = time.time()

            # Load data based on model
            if model_name == 'HKNNRF':
                X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = \
                    load_hknnrf_split(size, test_size=0.2, random_state=seed)

                # Train HKNNRF
                model = HKNNRFClassifier(
                    n_estimators=self.config['hyperparameters']['hknnrf']['n_estimators_range'][0],
                    max_depth=self.config['hyperparameters']['hknnrf']['max_depth_range'][0],
                    n_neighbors=self.config['hyperparameters']['hknnrf']['n_neighbors_range'][0],
                    random_state=seed
                )
                model.fit(X_train_rf, y_train_rf, X_train_knn, y_train_knn)
                y_pred = model.predict(X_test)

            else:
                # Standard train/test split
                X_train, X_test, y_train, y_test = load_multiclass_dataset(
                    size, test_size=0.2, random_state=seed
                )

                # Train model
                if model_name == 'SVM':
                    model = get_svm_model(**self.config['hyperparameters']['svm'])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif model_name == 'KNN':
                    model = get_knn_model(**self.config['hyperparameters']['knn'])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif model_name == 'RF':
                    model = get_random_forest_model(**self.config['hyperparameters']['random_forest'])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif model_name == 'SVMNB':
                    model = get_svmnb_model(**self.config['hyperparameters']['svmnb'], random_state=seed)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                elif model_name == 'MLP':
                    model = MLPClassifier(
                        num_classes=5,
                        **self.config['hyperparameters']['mlp'],
                        verbose=0,
                        random_state=seed,
                    )
                    self._fit_deep_model(model, X_train, y_train, seed)
                    y_pred = model.predict(X_test)

                elif model_name == 'CNN':
                    model = CNN1DClassifier(
                        num_classes=5,
                        **self.config['hyperparameters']['cnn'],
                        verbose=0,
                        random_state=seed,
                    )
                    self._fit_deep_model(model, X_train, y_train, seed)
                    y_pred = model.predict(X_test)

                else:
                    raise ValueError(f"Unknown model: {model_name}")

            training_time = time.time() - start_time

            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, 'multiclass')
            cm = compute_confusion_matrix(y_test, y_pred)

            # Package results
            result = {
                'model_name': model_name,
                'task_type': 'multiclass',
                'algorithm_pair': '5-class',
                'ciphertext_size': size,
                'train_size': len(y_train) if model_name != 'HKNNRF' else len(y_train_rf) + len(y_train_knn),
                'test_size': len(y_test),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'confusion_matrix': cm.tolist(),
                'training_time': training_time,
                'random_seed': seed,
                'timestamp': datetime.now().isoformat()
            }

            print(f"[OK] Accuracy: {metrics['accuracy']:.4f} | Time: {training_time:.2f}s")

            return result

        except Exception as e:
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_complete_matrix(self, models=None, sizes=None, binary_pairs=None):
        """
        Run complete experiment matrix.

        Args:
            models: List of models to test (default: all)
            sizes: List of sizes to test (default: all)
            binary_pairs: List of binary pairs (default: all 10)
        """
        if models is None:
            models = ['SVM', 'KNN', 'RF', 'HKNNRF', 'MLP', 'CNN', 'SVMNB']

        if sizes is None:
            sizes = self.config['ciphertext_sizes']

        if binary_pairs is None:
            binary_pairs = get_all_binary_pairs()

        print("\n" + "="*70)
        print("COMPLETE EXPERIMENT MATRIX")
        print("="*70)
        print(f"Models: {len(models)}")
        print(f"Sizes: {len(sizes)}")
        print(f"Binary pairs: {len(binary_pairs)}")
        print(f"Total binary experiments: {len(models) * len(sizes) * len(binary_pairs)}")
        print(f"Total multiclass experiments: {len(models) * len(sizes)}")
        print(f"Grand total: {len(models) * len(sizes) * (len(binary_pairs) + 1)}")
        print("="*70)

        # Run multiclass experiments
        print("\n>>> MULTICLASS EXPERIMENTS")
        for size in sizes:
            for model in models:
                result = self.run_multiclass_experiment(size, model)
                if result:
                    self.results.append(result)
                    if self.use_db:
                        self._save_to_db(result)

        # Run binary experiments
        print("\n>>> BINARY EXPERIMENTS")
        for size in sizes:
            # Convert size format for binary
            size_folder = size.lower().replace('kb', 'kb')
            for pair in binary_pairs:
                for model in models:
                    result = self.run_binary_experiment(pair, size_folder, model)
                    if result:
                        self.results.append(result)
                        if self.use_db:
                            self._save_to_db(result)

        # Save results to file
        self._save_results()

        print("\n" + "="*70)
        print(f"COMPLETE: {len(self.results)} experiments executed")
        print("="*70)

    def _save_to_db(self, result: Dict):
        """Save single result to database."""
        if not self.use_db:
            return

        try:
            self.db.insert_experiment(
                model_name=result['model_name'],
                task_type=result['task_type'],
                algorithm_pair=result['algorithm_pair'],
                ciphertext_size=result['ciphertext_size'],
                accuracy=result['accuracy'],
                precision=result['precision'],
                recall=result['recall'],
                f1_score=result['f1_score'],
                training_time=result['training_time'],
                hyperparameters={},
                confusion_matrix=np.array(result['confusion_matrix'])
            )
        except Exception as e:
            print(f"Warning: Database save failed: {e}")

    def _save_results(self):
        """Save all results to CSV and JSON."""
        if not self.results:
            return

        # Create DataFrame
        df = pd.DataFrame(self.results)

        # Remove confusion matrix for CSV (save separately)
        df_csv = df.drop(columns=['confusion_matrix'])

        # Save CSV
        csv_path = os.path.join(self.output_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df_csv.to_csv(csv_path, index=False)
        print(f"\n[SAVED] Results: {csv_path}")

        # Save JSON (with confusion matrices)
        json_path = os.path.join(self.output_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"[SAVED] Full results: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Run complete CipherBench experiments')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Models to test (default: all)')
    parser.add_argument('--sizes', nargs='+', default=None,
                       help='Sizes to test (default: all)')
    parser.add_argument('--task', choices=['binary', 'multiclass', 'all'], default='all',
                       help='Task type to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-db', action='store_true',
                       help='Skip database storage')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (512KB only, first 3 binary pairs)')

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(
        use_db=not args.no_db,
        random_seed=args.seed
    )

    # Determine experiment scope
    if args.quick:
        print("\n>>> QUICK TEST MODE")
        sizes = ['512KB']
        binary_pairs = get_all_binary_pairs()[:3]
    else:
        sizes = args.sizes
        binary_pairs = None

    # Run experiments
    if args.task == 'multiclass':
        print("\n>>> MULTICLASS ONLY")
        models = args.models or ['SVM', 'KNN', 'RF', 'HKNNRF', 'MLP', 'CNN', 'SVMNB']
        sizes = sizes or runner.config['ciphertext_sizes']
        for size in sizes:
            for model in models:
                result = runner.run_multiclass_experiment(size, model)
                if result:
                    runner.results.append(result)
                    if runner.use_db:
                        runner._save_to_db(result)
        runner._save_results()

    elif args.task == 'binary':
        print("\n>>> BINARY ONLY")
        models = args.models or ['SVM', 'KNN', 'RF', 'HKNNRF', 'MLP', 'CNN', 'SVMNB']
        sizes = sizes or runner.config['ciphertext_sizes']
        binary_pairs = binary_pairs or get_all_binary_pairs()
        for size in sizes:
            size_folder = size.lower().replace('kb', 'kb')
            for pair in binary_pairs:
                for model in models:
                    result = runner.run_binary_experiment(pair, size_folder, model)
                    if result:
                        runner.results.append(result)
                        if runner.use_db:
                            runner._save_to_db(result)
        runner._save_results()

    else:
        runner.run_complete_matrix(
            models=args.models,
            sizes=sizes,
            binary_pairs=binary_pairs
        )


if __name__ == '__main__':
    main()
