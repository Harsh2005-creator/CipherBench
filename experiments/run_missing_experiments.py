"""
Resume-capable experiment runner for CipherBench.

Loads the MOST RECENT consolidated results file under experiments/results/
(rather than a hard-coded historical filename that may no longer exist) and
runs only the experiments that are missing or flagged as suspicious.

A row is treated as "needing a rerun" (not just "missing") when it is a
binary-classification row with accuracy == 0.0, which is not a plausible
result on a perfectly-balanced binary test set. Existence of a row is
therefore not sufficient by itself to mark an experiment complete.
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import glob
import pandas as pd
import json
from datetime import datetime
from experiments.run_complete_experiments import ExperimentRunner
from src.utils.data_loader import get_all_binary_pairs

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')


def find_latest_results_file():
    """Find the most recent results_consolidated_*.csv file."""
    pattern = os.path.join(RESULTS_DIR, 'results_consolidated_*.csv')
    candidates = sorted(glob.glob(pattern))
    return candidates[-1] if candidates else None


def load_existing_results(results_file):
    """Load existing results from CSV file."""
    if results_file is None or not os.path.exists(results_file):
        print(f"No existing results found (looked for results_consolidated_*.csv in {RESULTS_DIR})")
        return pd.DataFrame()

    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} existing experiments from: {results_file}")
    return df


def is_experiment_valid(df, model, task, pair, size):
    """
    An experiment counts as done only if a matching row exists AND it does
    not show the signature of the known label-space bug (a binary-task row
    with accuracy exactly 0.0).
    """
    matches = df[
        (df['model_name'] == model) &
        (df['task_type'] == task) &
        (df['algorithm_pair'] == pair) &
        (df['ciphertext_size'] == size)
    ]
    if len(matches) == 0:
        return False

    row = matches.iloc[0]
    if task == 'binary' and pd.notna(row.get('accuracy')) and row['accuracy'] == 0.0:
        return False

    return True


def identify_missing_experiments(existing_df):
    """Identify all missing or suspicious experiments from the 330 required."""

    sizes_multiclass = ['1KB', '8KB', '64KB', '256KB', '512KB']
    models = ['SVM', 'KNN', 'RF', 'HKNNRF', 'MLP', 'CNN']
    binary_pairs = get_all_binary_pairs()

    missing = []

    for size in sizes_multiclass:
        for model in models:
            if not is_experiment_valid(existing_df, model, 'multiclass', '5-class', size):
                missing.append(('multiclass', '5-class', size, model))

    for size in sizes_multiclass:
        size_folder = size.lower()
        for pair in binary_pairs:
            for model in models:
                if not is_experiment_valid(existing_df, model, 'binary', pair, size_folder):
                    missing.append(('binary', pair, size_folder, model))

    return missing


def main():
    print("=" * 80)
    print("CIPHERBENCH RESUME RUNNER - MISSING / SUSPICIOUS EXPERIMENTS ONLY")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    existing_results_file = find_latest_results_file()
    existing_df = load_existing_results(existing_results_file)

    missing = identify_missing_experiments(existing_df)

    print(f"\n=== EXPERIMENT STATUS ===")
    print(f"Existing experiments (row present): {len(existing_df)}")
    print(f"Missing or suspicious (need rerun): {len(missing)}")
    print(f"Expected total: 330")
    print()

    if len(missing) == 0:
        print("All experiments complete and none flagged as suspicious. Nothing to run.")
        return

    print(f"=== EXPERIMENTS TO RUN ===")
    for i, (task, pair, size, model) in enumerate(missing[:10], 1):
        print(f"{i}. {task} | {pair} | {size} | {model}")
    if len(missing) > 10:
        print(f"... and {len(missing) - 10} more")
    print()

    response = input(f"Run {len(missing)} experiments? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    print()
    print("=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)
    print()

    runner = ExperimentRunner(
        output_dir=RESULTS_DIR,
        use_db=False,
        random_seed=42
    )

    completed = 0
    failed = 0

    for task, pair, size, model in missing:
        try:
            if task == 'multiclass':
                result = runner.run_multiclass_experiment(size, model)
            else:
                result = runner.run_binary_experiment(pair, size, model)

            if result:
                runner.results.append(result)
                completed += 1
            else:
                failed += 1
                print(f"[FAILED] {model} {task} {pair} {size}")

            if completed % 5 == 0:
                print(f"Progress: {completed}/{len(missing)} completed, {failed} failed")

        except Exception as e:
            print(f"[ERROR] {model} {task} {pair} {size}: {e}")
            failed += 1

    # Replace stale rows (if any) with fresh ones, keep everything else.
    existing_records = existing_df.to_dict('records') if not existing_df.empty else []
    rerun_keys = {(r['model_name'], r['task_type'], r['algorithm_pair'], r['ciphertext_size']) for r in runner.results}
    kept_records = [
        r for r in existing_records
        if (r['model_name'], r['task_type'], r['algorithm_pair'], r['ciphertext_size']) not in rerun_keys
    ]
    consolidated_results = kept_records + runner.results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_all = pd.DataFrame(consolidated_results)
    drop_cols = [c for c in ['confusion_matrix'] if c in df_all.columns]
    csv_path = os.path.join(RESULTS_DIR, f'results_consolidated_{timestamp}.csv')
    df_all.drop(columns=drop_cols).to_csv(csv_path, index=False)

    json_path = os.path.join(RESULTS_DIR, f'results_consolidated_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(consolidated_results, f, indent=2)

    print()
    print("=" * 80)
    print("RESUME EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experiments run this pass: {completed}")
    print(f"Failed: {failed}")
    print(f"Total consolidated: {len(consolidated_results)}")
    print()
    print(f"Consolidated results saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print()

    expected = 330
    actual = len(consolidated_results)
    print(f"Expected experiments: {expected}")
    print(f"Actual experiments: {actual}")

    if actual == expected:
        print("\n*** PRIMARY EXPERIMENT MATRIX COMPLETE ***")
    else:
        print(f"\nWARNING: Still missing {expected - actual} experiments")

    print("=" * 80)


if __name__ == '__main__':
    main()
