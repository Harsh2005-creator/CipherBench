# CipherBench

**Machine Learning Framework for Block Cipher Algorithm Identification**

CipherBench identifies which block cipher (AES, 3DES, Blowfish, CAST, or
RC2) produced a ciphertext, using only 10 NIST-randomness-derived
statistical features — no key or plaintext required. It reproduces the
HKNNRF method of Yuan et al. (2022) and extends it with two deep-learning
baselines (MLP, 1D-CNN), comparing all six models across binary (pairwise)
and five-class identification, at five ciphertext sizes (1KB–512KB).

For the full technical writeup — architecture, methodology, synopsis/paper
compliance, and reproducibility — see **[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)**.
For the verified experiment results, see **[docs/RESULTS.md](docs/RESULTS.md)**.
This README covers setup and day-to-day usage.

---

## Quick start

```bash
# 1. Install dependencies (Python 3.11+; developed on 3.14)
pip install -r requirements.txt

# 2. Verify the install
python tests/test_suite.py

# 3. Explore the project (no database required)
#    Run from the project root so the dark theme in .streamlit/config.toml is applied
streamlit run app/streamlit_app.py
```

The interface is styled by `app/style.css` (design tokens, cards, tabs, navigation)
and `.streamlit/config.toml` (base theme). Change colours or spacing there; no Python
edits are needed.

## Repository structure

```
CipherBench/
├── src/                      models/, utils/, train.py, demo.py
├── data/                     55 CSV datasets (binary/, multiclass/)
├── experiments/              experiment runners + results/
├── app/                      primary UI: streamlit_app.py + style.css + effects.js (no DB needed)
├── .streamlit/config.toml    dark theme for the app
├── api/                      optional Flask REST API (MySQL-backed)
├── database/                 MySQL schema + connection helpers (optional)
├── tests/                    test suite
├── docs/                     PROJECT_REPORT.md, RESULTS.md
└── archive/                  synopsis/paper reference extracts
```

## Usage

```bash
# Train one model on one task/size (binary trains all 10 pairs for that size)
python src/train.py --model hknnrf --task multiclass --size 512KB
python src/train.py --model svm --task binary --size 512KB

# Quick classical-models-only demo
python src/demo.py

# Run the full 330-experiment matrix (long-running)
python experiments/run_complete_experiments.py --no-db

# Verify / refresh the results file against the expected 330-row matrix
python experiments/run_missing_experiments.py

# Launch the app
streamlit run app/streamlit_app.py
```

Available `--model` values: `svm`, `knn`, `rf`, `hknnrf`, `mlp`, `cnn`, `all`.
Available `--size` values: `1KB`, `8KB`, `64KB`, `256KB`, `512KB`.

### Optional: MySQL store and REST API

```bash
export CIPHERBENCH_DB_PASSWORD=your_password    # never commit a real password
mysql -u root -p < database/schema.sql
python api/app.py
```

Neither is required to use the project — `app/streamlit_app.py` and all
training/experiment scripts work entirely from local files.

## Models

| Model | Type | Notes |
|-------|------|-------|
| SVM | Classical | `sklearn.svm.SVC`, linear kernel |
| KNN | Classical | `sklearn.neighbors.KNeighborsClassifier` |
| Random Forest | Classical | `sklearn.ensemble.RandomForestClassifier` |
| HKNNRF | Hybrid | Reproduces Yuan et al. (2022); RF leaf-index features + original features → KNN |
| MLP | Deep learning | 2 dense layers [64, 32], dropout, early stopping |
| 1D-CNN | Deep learning | Conv1D [32, 64], batch norm, max-pooling |

Deep-learning framework is **TensorFlow/Keras**, not PyTorch as listed in the
synopsis — PyTorch has no published wheel for the Python version used in
development. See [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) §6.

## Datasets

55 CSV files under `data/`: 5 multiclass (500 samples each) + 50 binary
(10 pairs × 5 sizes, 200 samples each), 10 NIST p-value features per sample.

## Results

The verified experiment matrix (330 experiments: 30 multiclass + 300
binary, all 6 models) lives in `experiments/results/results_consolidated_*.csv`
/ `.json`. See [docs/RESULTS.md](docs/RESULTS.md) for headline numbers and
how to reproduce them.

## Testing

```bash
python tests/test_suite.py
```

Covers dataset loading/integrity, all 6 models, metrics, confusion
matrices, data-leakage checks, and split reproducibility.

## Documentation

| Document | Contents |
|---|---|
| [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Architecture, methodology, synopsis/paper compliance, reproducibility, limitations |
| [docs/RESULTS.md](docs/RESULTS.md) | Verified result counts and headline numbers |

## Research paper

Yuan, K., Yu, D., Feng, J., Yang, L., Jia, C., Huang, Y. (2022). *A block
cipher algorithm identification scheme based on hybrid k-nearest neighbor
and random forest algorithm.* PeerJ Computer Science.
DOI [10.7717/peerj-cs.1110](https://doi.org/10.7717/peerj-cs.1110).
This project reproduces its HKNNRF methodology on a supplied pre-extracted
feature dataset; see [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) §7
for exactly what matches and what differs.

## Team

Minor Project, Semester 7 — Department of Information Technology,
Maharaja Surajmal Institute of Technology, New Delhi.
Sanyam Kumar, Aayush Ahuja, Harsh Ramrakhiani. Supervisor: Dr. Bharti Sharma.
