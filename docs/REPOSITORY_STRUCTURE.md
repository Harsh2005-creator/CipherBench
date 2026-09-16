# CipherBench Repository Structure

**Last Updated**: 2026-09-16  
**Status**: Organized and Production-Ready

---

## Directory Structure

```
CipherBench/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── config.yaml                  # Configuration settings
├── .gitignore                   # Git ignore rules
│
├── src/                         # Source code
│   ├── models/                  # ML models
│   │   ├── baseline.py          # SVM, KNN, RF models
│   │   ├── hknnrf.py            # Hybrid KNN+RF model
│   │   ├── mlp.py               # Multi-Layer Perceptron
│   │   └── cnn.py               # 1D-CNN model
│   │
│   ├── utils/                   # Utility functions
│   │   ├── data_loader.py       # Dataset loading
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── visualization.py     # Plotting utilities
│   │
│   ├── train.py                 # Main training script
│   └── demo.py                  # Quick demo script
│
├── data/                        # Datasets (55 CSV files)
│   ├── multiclass/              # 5-class datasets (5 files)
│   │   ├── 1KB.csv
│   │   ├── 8KB.csv
│   │   ├── 64KB.csv
│   │   ├── 256KB.csv
│   │   └── 512KB.csv
│   │
│   └── binary/                  # Pairwise datasets (50 files)
│       ├── 1kb/                 # 10 algorithm pairs
│       ├── 8kb/
│       ├── 64kb/
│       ├── 256kb/
│       └── 512kb/
│
├── experiments/                 # Experiment infrastructure
│   ├── run_complete_experiments.py  # Complete experiment runner
│   └── results/                 # Experiment outputs (CSV/JSON)
│
├── tests/                       # Test suite
│   └── test_suite.py            # 16 comprehensive tests
│
├── database/                    # Database integration
│   ├── schema.sql               # MySQL schema
│   ├── db_config.py             # Connection management
│   └── db_operations.py         # CRUD operations
│
├── api/                         # REST API
│   └── app.py                   # Flask API (7 endpoints)
│
├── dashboard/                   # Interactive dashboard
│   └── streamlit_app.py         # Streamlit app (4 tabs)
│
├── docs/                        # Documentation
│   ├── FINAL_AUDIT_REPORT.md            # Complete audit results
│   ├── PROJECT_STATUS_FINAL.md          # Final status report
│   ├── PAPER_REPLICATION_AUDIT.md       # Paper analysis
│   ├── IMPLEMENTATION_REPORT.md         # Detailed implementation
│   └── REPOSITORY_STRUCTURE.md          # This file
│
└── archive/                     # Reference materials
    ├── paper_extract.txt        # Research paper extract
    ├── synopsis_extract.txt     # Project synopsis
    └── legacy/                  # Old documentation
        ├── TIMELINE.md
        ├── PROJECT_STATUS.md
        ├── RESULTS_ANALYSIS.md
        ├── BINARY_CLASSIFICATION_RESULTS.md
        ├── COMPLETE_COMPARISON_PAPER_VS_IMPLEMENTATION.md
        ├── FINAL_RESULTS_SUMMARY.md
        ├── COMPLETION_SUMMARY.md
        └── EXECUTIVE_SUMMARY.md
```

---

## File Count Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Source Files** | 10 | Python implementation files |
| **Datasets** | 55 | CSV files (5 multiclass + 50 binary) |
| **Tests** | 1 | Comprehensive test suite (16 tests) |
| **Documentation** | 5 | Active documentation files |
| **Legacy Docs** | 8 | Archived historical documentation |
| **Config** | 2 | Configuration files |

**Total Active Files**: ~75 files

---

## Key Components

### Source Code (`src/`)
All Python source code organized by function:
- **Models**: 4 files implementing 6 ML models
- **Utils**: 3 files for data loading, metrics, visualization
- **Scripts**: 2 files for training and demo

### Data (`data/`)
55 validated CSV datasets:
- **Multiclass**: 5 files (500 samples each, 5 classes)
- **Binary**: 50 files (200 samples each, 2 classes)
- **Features**: 10 NIST-derived features per sample
- **Quality**: No NaN, no inf, perfectly balanced

### Experiments (`experiments/`)
Complete experiment infrastructure:
- **Runner**: Systematic execution of 330 possible experiments
- **Results**: CSV and JSON output storage
- **Reproducible**: Fixed seeds for consistency

### Tests (`tests/`)
Comprehensive validation:
- **16 tests** covering all components
- **100% pass rate**
- **Critical validations**: Architecture, data integrity, reproducibility

### Database (`database/`)
MySQL integration:
- **Schema**: Proper indexing and JSON storage
- **Operations**: Full CRUD functionality
- **Optional**: Can run without database using `--no-db`

### API (`api/`)
REST API for remote access:
- **7 endpoints** for querying results
- **Flask-based** with CORS enabled
- **JSON responses**

### Dashboard (`dashboard/`)
Interactive visualization:
- **4 tabs**: Comparison, Best Models, Confusion Matrices, Details
- **Filters**: Task, size, models
- **Export**: CSV download capability

### Documentation (`docs/`)
Professional technical documentation:
- **FINAL_AUDIT_REPORT.md**: Complete 18-phase audit
- **PROJECT_STATUS_FINAL.md**: Comprehensive status
- **PAPER_REPLICATION_AUDIT.md**: Paper analysis
- **IMPLEMENTATION_REPORT.md**: Detailed implementation
- **REPOSITORY_STRUCTURE.md**: This file

### Archive (`archive/`)
Reference materials and historical documentation:
- **Research materials**: Paper extract, synopsis
- **Legacy docs**: Historical reports preserved for reference

---

## Import Paths

All imports now use the `src.` prefix:

```python
# Correct imports
from src.models.hknnrf import HKNNRFClassifier
from src.utils.data_loader import load_multiclass_dataset
from src.utils.metrics import compute_metrics
```

**Note**: All files have been updated to use the new structure.

---

## Running Commands

### From Project Root

```bash
# Run tests
python tests/test_suite.py

# Train models
python src/train.py --model all --task multiclass --size 512KB

# Quick demo
python src/demo.py

# Run experiments
python experiments/run_complete_experiments.py --quick --no-db

# Launch dashboard
streamlit run dashboard/streamlit_app.py

# Launch API
python api/app.py
```

---

## Changes Made (2026-09-16)

### Organizational Changes
1. ✅ Created `src/` directory for all source code
2. ✅ Created `archive/` directory for reference materials
3. ✅ Created `archive/legacy/` for old documentation
4. ✅ Moved all source code to `src/`
5. ✅ Moved research materials to `archive/`
6. ✅ Moved legacy docs to `archive/legacy/`
7. ✅ Consolidated active docs in `docs/`
8. ✅ Updated all import paths

### Files Deleted
1. ✅ `=3.0.0` - Unknown artifact
2. ✅ `setup.bat` - Not needed
3. ✅ `run_full_hknnrf_test.py` - Replaced by test suite
4. ✅ `test_hknnrf_fix.py` - Replaced by test suite
5. ✅ All `__pycache__` directories - Cleaned

### Files Preserved
- ✅ All source code (now in `src/`)
- ✅ All datasets (55 files in `data/`)
- ✅ All active documentation (in `docs/`)
- ✅ All infrastructure (experiments, tests, database, api, dashboard)
- ✅ Configuration files (config.yaml, requirements.txt, .gitignore)

---

## Benefits of New Structure

### ✅ Professional Organization
- Clear separation of concerns
- Industry-standard `src/` layout
- Proper documentation hierarchy

### ✅ Easier Navigation
- Related files grouped together
- Less clutter in root directory
- Clear distinction between active and archived

### ✅ Better Maintainability
- Consistent import paths
- Organized documentation
- Clean git status

### ✅ Production Ready
- Suitable for deployment
- Easy to understand for new developers
- Professional appearance for submission

---

## For College Submission

This structure is ideal for:
- ✅ **Viva Demonstration**: Clear, organized, professional
- ✅ **Report Writing**: Well-documented with clear structure
- ✅ **Code Review**: Easy to navigate and understand
- ✅ **Future Work**: Easy to extend and maintain

---

**Repository Status**: ✅ **CLEAN, ORGANIZED, PRODUCTION-READY**

*Last reorganized: 2026-09-16 by Lead Software Engineer*
