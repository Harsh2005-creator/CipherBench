"""
Resume-capable experiment runner for CipherBench.
Loads existing results and runs only missing experiments.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from datetime import datetime
from experiments.run_complete_experiments import ExperimentRunner
from src.utils.data_loader import get_all_binary_pairs

def load_existing_results(results_file):
    """Load existing results from CSV file."""
    if not os.path.exists(results_file):
        print(f"No existing results found at: {results_file}")
        return pd.DataFrame()

    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} existing experiments from: {results_file}")
    return df

def is_experiment_completed(df, model, task, pair, size):
    """Check if an experiment already exists in results."""
    matches = df[
        (df['model_name'] == model) &
        (df['task_type'] == task) &
        (df['algorithm_pair'] == pair) &
        (df['ciphertext_size'] == size)
    ]
    return len(matches) > 0

def identify_missing_experiments(existing_df):
    """Identify all missing experiments from the 330 required."""

    sizes_multiclass = ['1KB', '8KB', '64KB', '256KB', '512KB']
    sizes_binary = ['1kb', '8kb', '64kb', '256kb', '512kb']
    models = ['SVM', 'KNN', 'RF', 'HKNNRF', 'MLP', 'CNN']
    binary_pairs = get_all_binary_pairs()

    missing = []

    # Check multiclass experiments
    for size in sizes_multiclass:
        for model in models:
            if not is_experiment_completed(existing_df, model, 'multiclass', '5-class', size):
                missing.append(('multiclass', '5-class', size, model))

    # Check binary experiments
    for size in sizes_binary:
        for pair in binary_pairs:
            for model in models:
                if not is_experiment_completed(existing_df, model, 'binary', pair, size):
                    missing.append(('binary', pair, size, model))

    return missing

def main():
    print("="*80)
    print("CIPHERBENCH RESUME RUNNER - MISSING EXPERIMENTS ONLY")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Path to existing results
    existing_results_file = 'experiments/results/results_20260916_223745.csv'

    # Load existing results
    existing_df = load_existing_results(existing_results_file)

    # Identify missing experiments
    missing = identify_missing_experiments(existing_df)

    print(f"\n=== EXPERIMENT STATUS ===")
    print(f"Existing experiments: {len(existing_df)}")
    print(f"Missing experiments: {len(missing)}")
    print(f"Expected total: 330")
    print()

    if len(missing) == 0:
        print("All experiments complete! Nothing to run.")
        return

    # Show what will be run
    print(f"=== MISSING EXPERIMENTS TO RUN ===")
    for i, (task, pair, size, model) in enumerate(missing[:10], 1):
        print(f"{i}. {task} | {pair} | {size} | {model}")
    if len(missing) > 10:
        print(f"... and {len(missing) - 10} more")
    print()

    # Confirm
    response = input(f"Run {len(missing)} missing experiments? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    print()
    print("="*80)
    print("RUNNING MISSING EXPERIMENTS")
    print("="*80)
    print()

    # Initialize runner
    runner = ExperimentRunner(
        output_dir='experiments/results',
        use_db=False,
        random_seed=42
    )

    # Run only missing experiments
    completed = 0
    failed = 0

    for task, pair, size, model in missing:
        try:
            if task == 'multiclass':
                result = runner.run_multiclass_experiment(size, model)
            else:  # binary
                result = runner.run_binary_experiment(pair, size, model)

            if result:
                runner.results.append(result)
                completed += 1
            else:
                failed += 1
                print(f"[FAILED] {model} {task} {pair} {size}")

            # Progress update
            if completed % 5 == 0:
                print(f"Progress: {completed}/{len(missing)} completed, {failed} failed")

        except Exception as e:
            print(f"[ERROR] {model} {task} {pair} {size}: {e}")
            failed += 1

    # Load existing results as dictionaries
    with open(existing_results_file.replace('.csv', '.json'), 'r') as f:
        existing_results = json.load(f)

    # Merge with new results
    consolidated_results = existing_results + runner.results

    # Save consolidated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    df_all = pd.DataFrame(consolidated_results)
    df_csv = df_all.drop(columns=['confusion_matrix'])
    csv_path = f'experiments/results/results_consolidated_{timestamp}.csv'
    df_csv.to_csv(csv_path, index=False)

    # Save JSON
    json_path = f'experiments/results/results_consolidated_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(consolidated_results, f, indent=2)

    print()
    print("="*80)
    print("RESUME EXECUTION COMPLETE")
    print("="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"New experiments run: {completed}")
    print(f"Failed: {failed}")
    print(f"Total consolidated: {len(consolidated_results)}")
    print()
    print(f"Consolidated results saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print()

    # Verify completion
    expected = 330
    actual = len(consolidated_results)
    print(f"Expected experiments: {expected}")
    print(f"Actual experiments: {actual}")

    if actual == expected:
        print("\n*** PRIMARY EXPERIMENT MATRIX COMPLETE ***")
    else:
        print(f"\nWARNING: Still missing {expected - actual} experiments")

    print("="*80)

if __name__ == '__main__':
    main()
