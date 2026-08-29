# CipherBench: Machine Learning Framework for Block Cipher Algorithm Identification

A comprehensive machine learning framework for identifying cryptographic algorithms from ciphertext using NIST-derived statistical features.

## 🎯 Overview

CipherBench implements and extends the HKNNRF (Hybrid K-Nearest Neighbors + Random Forest) methodology for cryptographic algorithm identification. The framework evaluates multiple machine learning approaches across:

- **5 Cipher Algorithms**: AES, 3DES, Blowfish, CAST, RC2
- **5 Ciphertext Sizes**: 1KB, 8KB, 64KB, 256KB, 512KB
- **2 Classification Tasks**: Binary (pairwise) and Multiclass (5-class)
- **6 Model Types**: SVM, KNN, Random Forest, HKNNRF, MLP, 1D-CNN

## 🏗️ Project Structure

```
Project/
├── data/
│   ├── binary/          # Binary classification datasets (10 algorithm pairs)
│   └── multiclass/      # Multiclass datasets (5 ciphertext sizes)
├── models/
│   ├── baseline.py      # SVM, KNN, Random Forest
│   ├── hknnrf.py        # Hybrid KNN+RF implementation
│   ├── mlp.py           # Multi-Layer Perceptron
│   └── cnn.py           # 1D Convolutional Neural Network
├── database/
│   ├── schema.sql       # MySQL database schema
│   ├── db_config.py     # Database connection
│   └── db_operations.py # CRUD operations
├── api/
│   └── app.py           # Flask REST API
├── dashboard/
│   └── streamlit_app.py # Streamlit visualization dashboard
├── utils/
│   ├── data_loader.py   # Dataset loading utilities
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Plotting utilities
├── train.py             # Main training script
├── requirements.txt     # Python dependencies
├── config.yaml          # Configuration file
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- MySQL 8.0+
- Git

### 2. Installation

```bash
# Clone or navigate to the project directory
cd "C:\Users\HARSH\Desktop\SEM  WORKS\SEM 7 All Work\Minor Project\Project"

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

**Configure MySQL credentials in `config.yaml`:**

```yaml
database:
  host: localhost
  user: root
  password: "YOUR_PASSWORD"  # Set your MySQL password
  database: cipherbench
  port: 3306
```

**Create the database:**

```bash
# Windows
mysql -u root -p < database/schema.sql

# Or manually in MySQL:
mysql -u root -p
CREATE DATABASE cipherbench;
USE cipherbench;
SOURCE database/schema.sql;
```

**Test database connection:**

```bash
python -c "from database.db_config import test_connection; test_connection()"
```

### 4. Train Models

**Train all models on multiclass 512KB dataset:**

```bash
python train.py --model all --task multiclass --size 512KB
```

**Train specific model:**

```bash
python train.py --model svm --task multiclass --size 512KB
python train.py --model hknnrf --task multiclass --size 512KB
python train.py --model mlp --task multiclass --size 512KB
```

**Train on all sizes:**

```bash
python train.py --model all --task multiclass --size all
```

**Options:**
- `--model`: `all`, `svm`, `knn`, `rf`, `hknnrf`, `mlp`, `cnn`
- `--task`: `all`, `binary`, `multiclass`
- `--size`: `all`, `1KB`, `8KB`, `64KB`, `256KB`, `512KB`

### 5. Launch Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

### 6. Start API Server (Optional)

```bash
python api/app.py
```

API documentation available at `http://localhost:5000/`

## 📊 Features

### Models Implemented

1. **SVM (Support Vector Machine)**: Linear kernel, gamma=0.001
2. **KNN (K-Nearest Neighbors)**: n_neighbors=3
3. **Random Forest**: 20 estimators
4. **HKNNRF (Hybrid KNN+RF)**: Two-stage ensemble (RF feature extraction + KNN classification)
5. **MLP (Multi-Layer Perceptron)**: Deep neural network with dropout
6. **1D-CNN**: Convolutional neural network for sequential feature analysis

### Evaluation Metrics

- Accuracy
- Precision (weighted for multiclass)
- Recall (weighted for multiclass)
- F1-Score
- Confusion Matrix
- Training Time

### Dashboard Features

- 📊 Model performance comparison
- 🎯 Best model ranking
- 📈 Confusion matrix visualization
- 📋 Detailed results table
- 📥 CSV export

### API Endpoints

- `GET /api/models` - List all models
- `GET /api/results?model=SVM&task=multiclass&size=512KB` - Query results
- `GET /api/best?task=multiclass&size=512KB&top_n=5` - Get top models
- `GET /api/confusion_matrix/<id>` - Get confusion matrix
- `GET /api/comparison?task=multiclass&size=512KB` - Compare all models
- `GET /api/stats` - Overall statistics

## 🔧 Configuration

Edit `config.yaml` to customize:

- Database credentials
- Model hyperparameters
- Data paths
- Algorithm label mappings

## 📝 Dataset Information

### NIST Features (10 features per sample)

- `aetPValue`: Approximate Entropy Test
- `custPValue`: Cumulative Sums Test
- `dtfPValue`: Discrete Fourier Transform Test
- `fwbtPValue`: Forward Backward Test
- `lrobPValue`: Linear Complexity Test
- `mtPValue`: Monobit Test
- `retPValue`: Random Excursions Test
- `revtPValue`: Random Excursions Variant Test
- `runsPValue`: Runs Test
- `stPValue`: Serial Test

### Label Mapping (Multiclass)

- 0: AES
- 1: 3DES
- 2: Blowfish
- 3: CAST
- 4: RC2

### Binary Classification Pairs

All 10 pairwise combinations of the 5 algorithms are available.

## 🧪 Testing

**Test data loader:**

```bash
python utils/data_loader.py
```

**Test metrics module:**

```bash
python utils/metrics.py
```

**Test database operations:**

```bash
python database/db_operations.py
```

## 📈 Expected Results

Based on the reference implementation:

- **512KB Multiclass**: HKNNRF typically achieves >95% accuracy
- **Larger ciphertext sizes** generally yield better performance
- **Deep learning models (MLP, CNN)** may achieve competitive or superior results

## 🐛 Troubleshooting

### Database Connection Error

```
Error: Access denied for user 'root'@'localhost'
```

**Solution**: Update `config.yaml` with correct MySQL password.

### TensorFlow Not Found

```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution**: Install TensorFlow:
```bash
pip install tensorflow
```

### Data File Not Found

```
FileNotFoundError: Dataset not found
```

**Solution**: Verify data files are in `data/binary/` and `data/multiclass/` directories.

### MySQL Database Does Not Exist

**Solution**: Run the schema file:
```bash
mysql -u root -p < database/schema.sql
```

## 🤝 Contributing

This project is part of an academic minor project. For questions or issues, contact the project team.

## 📄 License

Academic project - All rights reserved by the project team.

## 👥 Team

- Harsh Ramrakhiani
- Sanyam Kumar
- Aayush Ahuja

**Supervisor**: Dr. Bharti Sharma  
**Institution**: Maharaja Surajmal Institute of Technology, New Delhi

## 🔗 References

- Yuan et al. - Hybrid K-Nearest Neighbours and Random Forest methodology
- NIST Statistical Test Suite for Random and Pseudorandom Number Generators

## 📞 Support

For technical issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure MySQL is running and database is created
4. Check `config.yaml` for correct paths and credentials

---

**Built with**: Python, TensorFlow, scikit-learn, Flask, Streamlit, MySQL, matplotlib
