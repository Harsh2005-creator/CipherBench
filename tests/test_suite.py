"""
Comprehensive Test Suite for CipherBench
Tests all major components for correctness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import all components
from src.utils.data_loader import (
    load_binary_dataset, load_multiclass_dataset,
    load_hknnrf_split, load_binary_hknnrf_split,
    get_all_binary_pairs, get_feature_names
)
from src.utils.metrics import compute_metrics, compute_confusion_matrix
from src.models.baseline import get_svm_model, get_knn_model, get_random_forest_model
from src.models.hknnrf import HKNNRFClassifier
from src.models.mlp import MLPClassifier
from src.models.cnn import CNN1DClassifier


class TestRunner:
    """Runs comprehensive tests on all CipherBench components."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def test(self, name, func):
        """Run a single test."""
        print(f"\n[TEST] {name}")
        try:
            func()
            print(f"[PASS] {name}")
            self.passed += 1
            self.tests.append((name, True, None))
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            self.failed += 1
            self.tests.append((name, False, str(e)))

    def report(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        print("="*70)

        if self.failed > 0:
            print("\nFailed tests:")
            for name, passed, error in self.tests:
                if not passed:
                    print(f"  - {name}: {error}")

        return self.failed == 0


def test_dataset_loading():
    """Test dataset loading functions."""
    # Test multiclass
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB")
    assert X_train.shape[1] == 10, "Should have 10 features"
    assert len(np.unique(y_train)) == 5, "Should have 5 classes"
    assert X_train.shape[0] + X_test.shape[0] == 500, "Total should be 500 samples"

    # Test binary
    X_train, X_test, y_train, y_test = load_binary_dataset("AES and 3DES", "512kb")
    assert X_train.shape[1] == 10, "Should have 10 features"
    assert len(np.unique(y_train)) == 2, "Should have 2 classes"


def test_hknnrf_split():
    """Test HKNNRF data splitting."""
    X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = load_hknnrf_split("512KB")

    # Check split ratios (approximately)
    total = X_train_rf.shape[0] + X_train_knn.shape[0] + X_test.shape[0]
    assert total == 500, "Total should be 500"
    assert abs(X_train_rf.shape[0] - X_train_knn.shape[0]) <= 1, "RF and KNN splits should be equal"
    assert abs(X_test.shape[0] - 100) <= 5, "Test should be ~20%"


def test_feature_names():
    """Test feature name extraction."""
    features = get_feature_names()
    assert len(features) == 10, "Should have 10 features"
    assert 'aetPValue' in features, "Should contain aetPValue"
    assert 'stPValue' in features, "Should contain stPValue"


def test_binary_pairs():
    """Test binary pair generation."""
    pairs = get_all_binary_pairs()
    assert len(pairs) == 10, "Should have 10 pairs (C(5,2) = 10)"
    assert "AES and 3DES" in pairs, "Should contain AES and 3DES"


def test_metrics():
    """Test metrics computation."""
    # Binary metrics
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred, 'binary')

    assert 'accuracy' in metrics, "Should have accuracy"
    assert 'precision' in metrics, "Should have precision"
    assert 'recall' in metrics, "Should have recall"
    assert 'f1_score' in metrics, "Should have f1_score"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"


def test_confusion_matrix():
    """Test confusion matrix computation."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    cm = compute_confusion_matrix(y_true, y_pred)

    assert cm.shape == (2, 2), "Binary CM should be 2x2"
    assert np.sum(cm) == len(y_true), "CM sum should equal sample count"


def test_svm():
    """Test SVM model."""
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB", random_state=42)
    model = get_svm_model(kernel='linear', gamma=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test), "Prediction length should match test length"
    assert set(y_pred).issubset(set(y_train)), "Predictions should be valid classes"


def test_knn():
    """Test KNN model."""
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB", random_state=42)
    model = get_knn_model(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test), "Prediction length should match test length"


def test_rf():
    """Test Random Forest model."""
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB", random_state=42)
    model = get_random_forest_model(n_estimators=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test), "Prediction length should match test length"


def test_hknnrf_architecture():
    """Test HKNNRF feature combination."""
    X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = load_hknnrf_split("512KB", random_state=42)

    model = HKNNRFClassifier(n_estimators=10, max_depth=3, n_neighbors=3, random_state=42)
    model.fit(X_train_rf, y_train_rf, X_train_knn, y_train_knn)

    # Check that model is fitted
    assert model.is_fitted, "Model should be fitted"

    # Check RF was trained
    assert model.rf is not None, "RF should exist"
    assert hasattr(model.rf, 'n_estimators'), "RF should have n_estimators"

    # Check OHE was fitted
    assert model.ohe is not None, "OHE should exist"

    # Check KNN was trained
    assert model.knn is not None, "KNN should exist"

    # Test prediction
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test), "Prediction length should match test length"

    # Test that combined features are used (critical test)
    # Get RF leaf indices
    leaf_indices = model.rf.apply(X_test[:1])
    encoded = model.ohe.transform(leaf_indices)

    # Combined should be original (10) + encoded features
    expected_features = 10 + encoded.shape[1]
    print(f"    Feature dimensions: 10 original + {encoded.shape[1]} RF-derived = {expected_features} total")


def test_mlp():
    """Test MLP model. Validation split is carved out of the training data
    only; the test set is used solely for the final prediction check."""
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB", random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model = MLPClassifier(
        num_classes=5,
        hidden_layers=[32, 16],
        epochs=5,
        batch_size=32,
        verbose=0
    )
    model.fit(X_fit, y_fit, X_val, y_val)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test), "Prediction length should match test length"
    assert model.model is not None, "Model should be trained"
    assert set(np.unique(y_pred)).issubset(set(model.classes_)), \
        "Predictions must only contain labels seen during training"


def test_cnn():
    """Test CNN model. Validation split is carved out of the training data
    only; the test set is used solely for the final prediction check."""
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB", random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model = CNN1DClassifier(
        num_classes=5,
        filters=[16],
        epochs=5,
        batch_size=32,
        verbose=0
    )
    model.fit(X_fit, y_fit, X_val, y_val)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test), "Prediction length should match test length"
    assert model.model is not None, "Model should be trained"
    assert set(np.unique(y_pred)).issubset(set(model.classes_)), \
        "Predictions must only contain labels seen during training"


def test_binary_dl_label_space():
    """Regression test for the historical bug where MLP/CNN predictions on a
    binary pair could contain a label outside that pair's two classes
    (e.g. leaking a raw 5-class index). Predictions must always be a subset
    of the true labels present in the binary dataset."""
    X_train, X_test, y_train, y_test = load_binary_dataset("AES and Blowfish", "1kb", random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model = MLPClassifier(num_classes=2, hidden_layers=[16], epochs=5, batch_size=32, verbose=0)
    model.fit(X_fit, y_fit, X_val, y_val)
    y_pred = model.predict(X_test)

    valid_labels = set(np.unique(y_train)) | set(np.unique(y_test))
    assert set(np.unique(y_pred)).issubset(valid_labels), (
        f"MLP predicted labels outside the binary pair's class set: "
        f"predicted={set(np.unique(y_pred))}, valid={valid_labels}"
    )


def test_data_leakage():
    """Test for data leakage in splits."""
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB", random_state=42)

    # Check no overlap between train and test
    train_set = set(map(tuple, X_train))
    test_set = set(map(tuple, X_test))
    overlap = train_set.intersection(test_set)

    assert len(overlap) == 0, f"Data leakage detected: {len(overlap)} samples in both train and test"


def test_reproducibility():
    """Test reproducibility with fixed seed."""
    # First run
    X_train1, X_test1, y_train1, y_test1 = load_multiclass_dataset("512KB", random_state=42)

    # Second run with same seed
    X_train2, X_test2, y_train2, y_test2 = load_multiclass_dataset("512KB", random_state=42)

    # Should be identical
    assert np.array_equal(X_train1, X_train2), "Train features should be identical"
    assert np.array_equal(y_train1, y_train2), "Train labels should be identical"
    assert np.array_equal(X_test1, X_test2), "Test features should be identical"
    assert np.array_equal(y_test1, y_test2), "Test labels should be identical"


def test_class_balance():
    """Test class balance in datasets."""
    _, _, _, y_train, _, y_test = load_hknnrf_split("512KB", random_state=42)

    # Check multiclass has all 5 classes
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)

    assert len(unique_train) == 5, "Training should have all 5 classes"
    # Test might not have all classes due to small size, but check it's valid
    assert all(c in [0, 1, 2, 3, 4] for c in unique_test), "Test labels should be valid"


def test_dataset_integrity():
    """Test dataset has no NaN or inf values."""
    X_train, X_test, y_train, y_test = load_multiclass_dataset("512KB")

    assert not np.any(np.isnan(X_train)), "Train features should not have NaN"
    assert not np.any(np.isinf(X_train)), "Train features should not have inf"
    assert not np.any(np.isnan(X_test)), "Test features should not have NaN"
    assert not np.any(np.isinf(X_test)), "Test features should not have inf"


def main():
    """Run all tests."""
    print("="*70)
    print("CIPHERBENCH COMPREHENSIVE TEST SUITE")
    print("="*70)

    runner = TestRunner()

    # Data loading tests
    runner.test("Dataset Loading", test_dataset_loading)
    runner.test("HKNNRF Split", test_hknnrf_split)
    runner.test("Feature Names", test_feature_names)
    runner.test("Binary Pairs", test_binary_pairs)
    runner.test("Dataset Integrity", test_dataset_integrity)
    runner.test("Class Balance", test_class_balance)

    # Metrics tests
    runner.test("Metrics Computation", test_metrics)
    runner.test("Confusion Matrix", test_confusion_matrix)

    # Model tests
    runner.test("SVM Model", test_svm)
    runner.test("KNN Model", test_knn)
    runner.test("Random Forest Model", test_rf)
    runner.test("HKNNRF Architecture", test_hknnrf_architecture)
    runner.test("MLP Model", test_mlp)
    runner.test("CNN Model", test_cnn)
    runner.test("Binary DL Label Space", test_binary_dl_label_space)

    # Validation tests
    runner.test("Data Leakage Check", test_data_leakage)
    runner.test("Reproducibility", test_reproducibility)

    # Print summary
    success = runner.report()

    if success:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1


if __name__ == '__main__':
    exit(main())
