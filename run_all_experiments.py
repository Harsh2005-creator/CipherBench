"""
Run ALL required experiments for CipherBench (330 total)

This script executes the complete experiment matrix:
- 30 multiclass experiments (5 sizes × 6 models)
- 300 binary experiments (10 pairs × 5 sizes × 6 models)

With proper test set isolation for MLP/CNN.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from datetime import datetime
from experiments.run_complete_experiments import ExperimentRunner
from src.utils.data_loader import get_all_binary_pairs

def main():
    print("="*80)
    print("CIPHERBENCH COMPLETE EXPERIMENT EXECUTION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize runner
    runner = ExperimentRunner(
        output_dir='experiments/results',
        use_db=False,  # Disable DB for now
        random_seed=42
    )

    # Define experiment matrix
    sizes = ['1KB', '8KB', '64KB', '256KB', '512KB']
    models = ['SVM', 'KNN', 'RF', 'HKNNRF', 'MLP', 'CNN']
    binary_pairs = get_all_binary_pairs()

    print(f"Sizes: {len(sizes)}")
    print(f"Models: {len(models)}")
    print(f"Binary pairs: {len(binary_pairs)}")
    print()
    print(f"Total multiclass: {len(sizes)} × {len(models)} = {len(sizes) * len(models)}")
    print(f"Total binary: {len(binary_pairs)} × {len(sizes)} × {len(models)} = {len(binary_pairs) * len(sizes) * len(models)}")
    print(f"Grand total: {len(sizes) * len(models) + len(binary_pairs) * len(sizes) * len(models)} experiments")
    print()
    print("="*80)
    print()

    # Track progress
    total_experiments = len(sizes) * len(models) + len(binary_pairs) * len(sizes) * len(models)
    completed = 0
    failed = 0

    # Run multiclass experiments
    print(">>> PHASE 1: MULTICLASS EXPERIMENTS")
    print()
    for size in sizes:
        for model in models:
            try:
                result = runner.run_multiclass_experiment(size, model)
                if result:
                    runner.results.append(result)
                    completed += 1
                else:
                    failed += 1
                print(f"Progress: {completed}/{total_experiments} completed, {failed} failed")
            except Exception as e:
                print(f"[ERROR] {model} {size}: {e}")
                failed += 1

    print()
    print("="*80)
    print()

    # Run binary experiments
    print(">>> PHASE 2: BINARY EXPERIMENTS")
    print()
    for size in sizes:
        size_folder = size.lower()
        for pair in binary_pairs:
            for model in models:
                try:
                    result = runner.run_binary_experiment(pair, size_folder, model)
                    if result:
                        runner.results.append(result)
                        completed += 1
                    else:
                        failed += 1

                    # Progress update every 10 experiments
                    if completed % 10 == 0:
                        print(f"Progress: {completed}/{total_experiments} completed, {failed} failed")
                except Exception as e:
                    print(f"[ERROR] {model} {pair} {size}: {e}")
                    failed += 1

    # Save all results
    runner._save_results()

    print()
    print("="*80)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: {completed}/{total_experiments}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/total_experiments*100:.1f}%")
    print()
    print(f"Results saved to: {runner.output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
