"""
Quick demo script - Baseline models only (no deep learning).
Trains SVM, KNN, RF, and HKNNRF on 512KB multiclass data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_multiclass_dataset, load_hknnrf_split, load_config
from utils.metrics import compute_metrics, compute_confusion_matrix
from models.baseline import get_svm_model, get_knn_model, get_random_forest_model
from models.hknnrf import HKNNRFClassifier
import time


def main():
    print("="*60)
    print("CipherBench Quick Demo - Baseline Models")
    print("="*60)
    print("\nTraining SVM, KNN, RF, and HKNNRF on 512KB multiclass data")
    print("(Deep learning models skipped - requires TensorFlow)")
    print()

    # Load configuration
    config = load_config()

    # Load data
    print("[1/5] Loading data...")
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB")
    print(f"✓ Loaded: Train={X_train.shape}, Test={X_test.shape}")

    results = {}

    # Train SVM
    print("\n[2/5] Training SVM...")
    start = time.time()
    svm = get_svm_model(**config['hyperparameters']['svm'])
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_time = time.time() - start

    metrics_svm = compute_metrics(y_test, y_pred_svm, 'multiclass')
    results['SVM'] = metrics_svm
    print(f"✓ SVM Accuracy: {metrics_svm['accuracy']:.4f} (trained in {svm_time:.2f}s)")

    # Train KNN
    print("\n[3/5] Training KNN...")
    start = time.time()
    knn = get_knn_model(**config['hyperparameters']['knn'])
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_time = time.time() - start

    metrics_knn = compute_metrics(y_test, y_pred_knn, 'multiclass')
    results['KNN'] = metrics_knn
    print(f"✓ KNN Accuracy: {metrics_knn['accuracy']:.4f} (trained in {knn_time:.2f}s)")

    # Train Random Forest
    print("\n[4/5] Training Random Forest...")
    start = time.time()
    rf = get_random_forest_model(**config['hyperparameters']['random_forest'])
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_time = time.time() - start

    metrics_rf = compute_metrics(y_test, y_pred_rf, 'multiclass')
    results['RF'] = metrics_rf
    print(f"✓ RF Accuracy: {metrics_rf['accuracy']:.4f} (trained in {rf_time:.2f}s)")

    # Train HKNNRF
    print("\n[5/5] Training HKNNRF...")
    print("   Loading split data for HKNNRF...")
    X_train_rf, X_train_knn, X_test_h, y_train_rf, y_train_knn, y_test_h = load_hknnrf_split("512KB")

    start = time.time()
    hknnrf = HKNNRFClassifier(n_estimators=85, max_depth=5, n_neighbors=7)
    hknnrf.fit(X_train_rf, y_train_rf, X_train_knn, y_train_knn)
    y_pred_hknnrf = hknnrf.predict(X_test_h)
    hknnrf_time = time.time() - start

    metrics_hknnrf = compute_metrics(y_test_h, y_pred_hknnrf, 'multiclass')
    results['HKNNRF'] = metrics_hknnrf
    print(f"✓ HKNNRF Accuracy: {metrics_hknnrf['accuracy']:.4f} (trained in {hknnrf_time:.2f}s)")

    # Summary
    print("\n" + "="*60)
    print("Results Summary (512KB Multiclass)")
    print("="*60)
    print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*60)

    for model_name, metrics in results.items():
        print(f"{model_name:<10} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")

    print("="*60)

    # Determine best model
    accuracies = {name: m['accuracy'] for name, m in results.items()}
    best_model = max(accuracies, key=accuracies.get)
    print(f"\n🏆 Best Model: {best_model} with {accuracies[best_model]:.4f} accuracy")

    print("\n✓ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Set MySQL password in config.yaml")
    print("2. Create database: mysql -u root -p < database/schema.sql")
    print("3. Run training: python train.py --model svm --task multiclass --size 512KB")
    print("4. View results: python -c \"from database.db_operations import ExperimentDB; db=ExperimentDB(); print(db.get_experiments())\"")
    print("\nNote: Deep learning models (MLP, CNN) require TensorFlow compatible with Python 3.14")


if __name__ == "__main__":
    main()
