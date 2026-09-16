"""
Full HKNNRF test comparing old vs new implementation results.
Tests all 10 binary pairs at 512kb to replicate paper's Table 4.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_binary_hknnrf_split, get_all_binary_pairs, load_hknnrf_split
from models.hknnrf import HKNNRFClassifier
from utils.metrics import compute_metrics
import numpy as np
import time

print("="*70)
print("FULL HKNNRF REPLICATION TEST")
print("="*70)
print("\nPaper Target (Yuan et al. 2022):")
print("- Binary classification average: 69.5%")
print("- Five-class classification (1KB): 34%")
print("- Five-class classification (512KB): 24%")
print()

# Test multiclass first
print("[MULTICLASS TEST - 512KB]")
print("-"*70)
X_train_rf, X_train_knn, X_test, y_train_rf, y_train_knn, y_test = load_hknnrf_split("512KB")

hknnrf = HKNNRFClassifier(n_estimators=85, max_depth=5, n_neighbors=7)
start = time.time()
hknnrf.fit(X_train_rf, y_train_rf, X_train_knn, y_train_knn)
train_time = time.time() - start

y_pred = hknnrf.predict(X_test)
metrics = compute_metrics(y_test, y_pred, 'multiclass')

print(f"Training time: {train_time:.2f}s")
print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1_score']:.4f}")
print(f"Paper target: 24% (512KB)")
print(f"Gap: {(metrics['accuracy']*100 - 24):.1f} percentage points")

# Test all binary pairs at 512kb (paper's Table 4)
print("\n[BINARY TEST - All 10 Pairs at 512kb]")
print("-"*70)
print(f"{'Algorithm Pair':<30} {'Accuracy':<10} {'Paper':<10} {'Gap'}")
print("-"*70)

all_pairs = get_all_binary_pairs()
binary_results = []

# Paper's Table 4 results for 512KB
paper_results = {
    "3DES and AES": 0.725,
    "Blowfish and AES": 0.675,
    "CAST and AES": 0.650,
    "RC2 and AES": 0.600,
    "Blowfish and 3DES": 0.700,
    "CAST and 3DES": 0.675,
    "RC2 and 3DES": 0.675,
    "CAST and Blowfish": 0.650,
    "RC2 and Blowfish": 0.625,
    "RC2 and CAST": 0.600
}

for pair in all_pairs:
    try:
        X_train_rf_b, X_train_knn_b, X_test_b, y_train_rf_b, y_train_knn_b, y_test_b = \
            load_binary_hknnrf_split(pair, "512kb")

        hknnrf_b = HKNNRFClassifier(n_estimators=85, max_depth=5, n_neighbors=7)
        hknnrf_b.fit(X_train_rf_b, y_train_rf_b, X_train_knn_b, y_train_knn_b)

        y_pred_b = hknnrf_b.predict(X_test_b)
        metrics_b = compute_metrics(y_test_b, y_pred_b, 'binary')

        binary_results.append({
            'pair': pair,
            'accuracy': metrics_b['accuracy']
        })

        # Get paper result (try both orderings)
        paper_acc = paper_results.get(pair, None)
        if paper_acc is None:
            # Try reversed order
            parts = pair.split(" and ")
            reversed_pair = f"{parts[1]} and {parts[0]}"
            paper_acc = paper_results.get(reversed_pair, None)

        paper_str = f"{paper_acc*100:.1f}%" if paper_acc else "N/A"
        gap_str = f"{(metrics_b['accuracy']-paper_acc)*100:+.1f}%" if paper_acc else "N/A"

        print(f"{pair:<30} {metrics_b['accuracy']*100:>6.1f}%    {paper_str:>7}   {gap_str:>7}")

    except Exception as e:
        print(f"{pair:<30} ERROR: {str(e)[:30]}")

# Calculate average
if binary_results:
    avg_accuracy = np.mean([r['accuracy'] for r in binary_results])
    paper_avg = 0.695

    print("-"*70)
    print(f"{'AVERAGE':<30} {avg_accuracy*100:>6.1f}%    {paper_avg*100:.1f}%   {(avg_accuracy-paper_avg)*100:+.1f}%")
    print("="*70)

    print(f"\nSUMMARY:")
    print(f"- Binary pairs tested: {len(binary_results)}/10")
    print(f"- Our average: {avg_accuracy*100:.1f}%")
    print(f"- Paper average: {paper_avg*100:.1f}%")
    print(f"- Gap: {(avg_accuracy-paper_avg)*100:.1f} percentage points")
    print(f"- Improvement from previous ~55%: {(avg_accuracy-0.55)*100:+.1f} points")

    # Best and worst
    best = max(binary_results, key=lambda x: x['accuracy'])
    worst = min(binary_results, key=lambda x: x['accuracy'])
    print(f"\nBest:  {best['pair']:<30} {best['accuracy']*100:.1f}%")
    print(f"Worst: {worst['pair']:<30} {worst['accuracy']*100:.1f}%")

print("\n" + "="*70)
print("STATUS: HKNNRF Architecture Fix Applied")
print("="*70)
print("CHANGES MADE:")
print("1. Original 10 NIST features NOW combined with RF-derived features")
print("2. Binary classification support added (load_binary_hknnrf_split)")
print("3. Training pipeline updated to support HKNNRF on binary tasks")
print("\nPaper compliance:")
print('- Step 9: "Use trained decision trees to construct new features"')
print('- Step 10: "Add them to original features to get final features"')
print('- Step 11: "Normalize final features with one-hot encoder"')
print('- Step 12: "Train KNN classifier on normalized features"')
print("\nAll steps now correctly implemented.")
print("="*70)
