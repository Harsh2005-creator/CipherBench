# Results

**Source**: `experiments/results/results_consolidated_20260918_002852.csv`
(+ matching `.json` with full confusion matrices) — 385 experiments: 35
five-class + 350 binary (10 pairs × 5 sizes), for all 7 models. Regenerate
with `python experiments/run_complete_experiments.py --no-db`.

## Coverage

| | Multiclass | Binary | Total |
|---|---|---|---|
| Expected | 35 | 350 | 385 |
| Present | 35 | 350 | 385 |

Every model has exactly 55 rows (5 multiclass + 50 binary).

## Multiclass (5-class), mean accuracy across all 5 sizes

| Model | Mean accuracy |
|---|---|
| KNN | 0.210 |
| SVMNB | 0.204 |
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

### SVM + Naive Bayes (extension)

SVMNB, an RBF SVM and Gaussian Naive Bayes soft-voted on the same ten
features, scores 0.204 five-class (second to KNN's 0.210, level with RF's
0.202) and 0.486 binary (the lowest binary mean of the seven, within noise of
chance). It is included as an extension; it does not outperform the other
models on this split.

## Binary, mean accuracy across all 10 pairs × 5 sizes

| Model | Mean accuracy |
|---|---|
| HKNNRF | 0.515 |
| MLP | 0.512 |
| CNN | 0.507 |
| KNN | 0.505 |
| RF | 0.505 |
| SVM | 0.498 |
| SVMNB | 0.486 |

All seven models cluster around 50% (chance level for a balanced 2-class
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
| SVMNB | 0.500 | 0.550 | 0.600 | 0.500 | 0.575 |

## Reproducing these numbers

```bash
python experiments/run_complete_experiments.py --no-db
```

Each experiment uses one seeded, stratified 80/20 split. With 100 (five-class)
or 40 (binary) test samples, one sample moves accuracy by 1 to 2.5 points, so
differences of that size between models are not meaningful. MLP/CNN numbers
may also shift run-to-run on different hardware (see
`docs/PROJECT_REPORT.md` §9).
