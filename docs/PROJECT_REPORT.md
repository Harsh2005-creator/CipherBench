# CipherBench — Project Report

## 1. Overview

CipherBench identifies which block cipher (AES, 3DES, Blowfish, CAST, or
RC2) produced a ciphertext, using 10 NIST-randomness-derived statistical
features of the ciphertext itself — no key or plaintext required. It
reproduces the **HKNNRF** method of Yuan et al. (2022) and compares it
against SVM, KNN, and Random Forest baselines, then extends the comparison
with two deep-learning models (MLP, 1D-CNN) and an SVM + Naive Bayes ensemble. All seven models are evaluated
on two tasks — binary (every algorithm pair) and five-class — across five
ciphertext sizes (1KB–512KB).

## 2. Repository structure

```
CipherBench/
├── README.md, requirements.txt, config.yaml, .gitignore
├── src/
│   ├── models/          baseline.py (SVM/KNN/RF), hknnrf.py, svmnb.py, mlp.py, cnn.py
│   ├── utils/            data_loader.py, metrics.py
│   ├── train.py           single-run CLI trainer
│   └── demo.py             quick classical-models-only demo
├── data/                  55 CSV datasets (binary/, multiclass/)
├── experiments/
│   ├── run_complete_experiments.py   experiment runner (all training logic lives here)
│   └── results/            results_consolidated_*.csv / .json
├── app/                    primary UI — streamlit_app.py (logic) + style.css + effects.js (design), no database needed
├── .streamlit/config.toml  dark theme (colours, font) picked up when launched from the project root
├── api/app.py               optional Flask REST API (MySQL-backed)
├── database/                 MySQL schema + connection helpers (optional)
├── tests/test_suite.py       test suite
├── docs/                     this report + docs/RESULTS.md
└── archive/                  synopsis and paper reference extracts
```

The Streamlit app reads the result files directly, so it has no external
service dependency. `api/app.py` and the `database/` package are optional —
the API returns a clear 503 message rather than crashing if no MySQL
database is configured.

## 3. Data pipeline

```
data/*.csv → src/utils/data_loader.py (stratified, seeded 80/20 split)
           → model (SVM / KNN / RF / HKNNRF / MLP / CNN / SVMNB)
           → src/utils/metrics.py (accuracy, precision, recall, F1, confusion matrix)
           → experiments/results/*.csv, *.json
           → app/streamlit_app.py
```

**Datasets**: 55 CSV files — 5 multiclass (500 samples each) + 50 binary
(10 pairs × 5 sizes, 200 samples each). Every load validates column count,
NaN, and inf values. Splits use `sklearn.train_test_split(stratify=y,
random_state=seed)`, so results are reproducible for a fixed seed.

**Features**: 10 NIST randomness-test p-values per sample (`aetPValue`,
`custPValue`, `dtfPValue`, `fwbtPValue`, `lrobPValue`, `mtPValue`,
`retPValue`, `revtPValue`, `runsPValue`, `stPValue`), used as-is — no
scaling or feature selection, matching the base paper's described
methodology.

## 4. Models

| Model | Implementation | Notes |
|---|---|---|
| SVM | `sklearn.svm.SVC`, linear kernel | classical baseline |
| KNN | `sklearn.neighbors.KNeighborsClassifier` | classical baseline |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | classical baseline |
| HKNNRF | `src/models/hknnrf.py` | RF leaf-index features one-hot encoded and **concatenated with the original 10 features**, then classified with KNN (paper Steps 9–12) |
| MLP | `src/models/mlp.py` | Keras `Sequential`, 2 dense layers [64, 32] + dropout |
| 1D-CNN | `src/models/cnn.py` | Keras `Sequential`, Conv1D [32, 64] + batch norm + max-pool |
| SVM + Naive Bayes (SVMNB) | `src/models/svmnb.py` | extension: soft-voting ensemble of a standardised RBF SVM and a Gaussian Naive Bayes model |

**Evaluation protocol** (identical for every model): an 80/20
train/test split is taken once per experiment; for MLP/CNN, a further
80/20 split of the *training* portion provides the validation set used for
early stopping (`EarlyStopping(monitor="val_loss", patience=15,
restore_best_weights=True)`). The held-out test set is used exactly once,
for the final `predict()` call — never for training or model selection.
MLP/CNN additionally call `keras.utils.set_random_seed()` before building
each model for run-to-run reproducibility.

## 5. Experiment matrix

```
Multiclass: 5 sizes × 7 models                    =  35 experiments
Binary:     10 pairs × 5 sizes × 7 models          = 350 experiments
                                             Total  = 385 experiments
```

The current results file, `experiments/results/results_consolidated_*.csv`
(see `docs/RESULTS.md`), contains all 385 rows, produced by
`experiments/run_complete_experiments.py` with a fixed seed (42) per
experiment and the evaluation protocol above.

## 6. Synopsis compliance

| Requirement | Status |
|---|---|
| 5 ciphers, 5 ciphertext sizes | Met |
| Binary (all 10 pairs) + five-class identification | Met |
| Internal 80/20 split, further split for HKNNRF's RF/KNN stages | Met |
| SVM, KNN, RF baselines + HKNNRF reproduction | Met |
| MLP and 1D-CNN extension | Met |
| Additional extension: SVM + Naive Bayes ensemble | Added |
| Accuracy / precision / recall / F1 / confusion matrices | Met |
| Deep-learning framework: PyTorch | **Deviation** — implemented in TensorFlow/Keras (`tf-nightly`) instead, because PyTorch has no published wheel for the Python version used in this environment |
| Results stored in MySQL, exposed via Flask API and Streamlit | Met, as optional components — the primary UI works from local result files without a database |
| Multi-seed stability / cross-ciphertext-size generalization | Not implemented — see §8 |

## 7. Comparison with the base paper

Yuan, K., Yu, D., Feng, J., Yang, L., Jia, C., Huang, Y. (2022). *A block
cipher algorithm identification scheme based on hybrid k-nearest neighbor
and random forest algorithm.* PeerJ Computer Science. DOI 10.7717/peerj-cs.1110.

The HKNNRF pipeline (RF leaf-index feature extraction → one-hot encoding →
concatenation with the original features → KNN classification) matches the
paper's Steps 9–12. The paper reports an average binary accuracy of 69.5%
and a five-class accuracy range of 24–34% for HKNNRF; this project's
results (`docs/RESULTS.md`) are lower across all seven models, including
HKNNRF. An ANOVA test on the 10 supplied NIST p-value features found none
of them statistically significant for distinguishing algorithms — this
project consumes a pre-extracted feature dataset rather than generating its
own ciphertext, and the feature set available here appears to carry less
discriminative signal than whatever the paper's authors used. This is
reported as a dataset/feature-representation constraint, not an
implementation defect — every model shows the same ceiling, not just
HKNNRF.

## 8. Known limitations

- **Framework**: TensorFlow/Keras, not PyTorch (see §6).
- **No ciphertext-generation pipeline**: the project works from the
  pre-extracted NIST-feature CSVs supplied for it, not from raw ciphertext.
- **Feature discriminability**: caps achievable accuracy for every model
  (see §7); not something a different model architecture can fix.
- **No multi-seed stability analysis or cross-ciphertext-size
  generalization** experiments (train on one size, test on another) —
  the synopsis mentions these as part of the analysis arm; the verified
  matrix here uses one seed per experiment and trains/tests on matching
  sizes only.
- **MLP/CNN determinism**: seeded, but not guaranteed bit-exact across
  different CPU hardware (floating-point non-associativity).
- **Database/API are optional**: `api/app.py` requires a reachable MySQL instance
  and returns a clear 503 message instead of crashing when one isn't
  configured.

## 9. Reproducibility

```bash
pip install -r requirements.txt
python tests/test_suite.py                        # verify the install
python src/train.py --model hknnrf --task multiclass --size 512KB
python src/demo.py                                 # quick classical-models demo
python experiments/run_complete_experiments.py --no-db   # full 385-experiment matrix
streamlit run app/streamlit_app.py                 # explore everything, no DB needed
```

Database credentials are never hard-coded: `config.yaml` ships with an
empty password placeholder, and `database/db_config.py` reads
`CIPHERBENCH_DB_PASSWORD` from the environment when the optional MySQL
integration is used:

```bash
export CIPHERBENCH_DB_PASSWORD=your_password
mysql -u root -p < database/schema.sql
python api/app.py
```

All paths in the codebase are resolved relative to the project root (via
`__file__`), so the project runs the same regardless of the working
directory it's launched from or the machine it's cloned to.

## 10. Testing

```bash
python tests/test_suite.py
```

Covers dataset loading and integrity, all 7 models, metrics and confusion
matrix computation, train/test leakage checks, and split reproducibility.
