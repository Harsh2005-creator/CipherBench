# Results

**Source**: `experiments/results/results_consolidated_20260918_002852.csv`
(+ matching `.json` with full confusion matrices) — 330 experiments: 30
five-class + 300 binary (10 pairs × 5 sizes), for all 6 models. Regenerate
with `python experiments/run_complete_experiments.py --no-db`.

## Coverage

| | Multiclass | Binary | Total |
|---|---|---|---|
| Expected | 30 | 300 | 330 |
| Present | 30 | 300 | 330 |

Every model has exactly 55 rows (5 multiclass + 50 binary).

## Multiclass (5-class), mean accuracy across all 5 sizes

| Model | Mean accuracy |
|---|---|
| KNN | 0.210 |
| RF | 0.202 |
| MLP | 0.186 |
| CNN | 0.182 |
| HKNNRF | 0.164 |
| SVM | 0.130 |

HKNNRF's best single result is 0.25 at 256KB; its range across sizes is
0.11–0.25. All models sit within a 0.13–0.25 band, consistent with the
paper's own reported range for the baselines (KNN 21%, RF 22%, SVM 23%,
HKNNRF 24–34%) — see `docs/PROJECT_REPORT.md` §7 for the feature-quality
discussion behind why the ceiling sits well under 100%.

## Binary, mean accuracy across all 10 pairs × 5 sizes

| Model | Mean accuracy |
|---|---|
| HKNNRF | 0.515 |
| MLP | 0.512 |
| CNN | 0.507 |
| KNN | 0.505 |
| RF | 0.505 |
| SVM | 0.498 |

All six models cluster around 50% (chance level for a balanced 2-class
problem). HKNNRF has the highest mean, consistent with it being the paper's
proposed method, though the margin over the other models here is modest.

### AES vs. 3DES

The paper's headline pair. Across all five sizes:

| Model | 1KB | 8KB | 64KB | 256KB | 512KB |
|---|---|---|---|---|---|
| SVM | 0.350 | 0.550 | 0.525 | 0.425 | 0.500 |
| KNN | 0.425 | 0.475 | 0.450 | 0.525 | 0.550 |
| RF | 0.575 | 0.400 | 0.575 | 0.450 | 0.425 |
| HKNNRF | 0.475 | 0.525 | 0.525 | 0.500 | 0.475 |
| MLP | 0.350 | 0.450 | 0.525 | 0.500 | 0.600 |
| CNN | 0.525 | 0.475 | 0.525 | 0.500 | 0.550 |

## Reproducing these numbers

```bash
python experiments/run_complete_experiments.py --no-db
```

Classical models (SVM/KNN/RF/HKNNRF) reproduce exactly given the fixed
seed. MLP/CNN numbers may shift by a few points run-to-run on different
hardware (see `docs/PROJECT_REPORT.md` §9) but stay within the bands shown
above.
