# CRITICAL ISSUES AUDIT - CipherBench

**Date**: 2026-09-16  
**Status**: INCOMPLETE - Major Issues Found

---

## 🚨 CRITICAL ISSUE #1: TEST SET LEAKAGE IN MLP/CNN

**Location**: 
- `src/models/mlp.py` line 91
- `src/models/cnn.py` line 118

**Problem**:
```python
# WRONG - Test set used for validation during training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),  # ❌ TEST SET LEAKAGE
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=verbose
)
```

**Impact**: 
- Early stopping uses test set performance
- Model selection biased toward test set
- Results are INVALID for MLP and CNN
- This violates fundamental ML evaluation principles

**Required Fix**:
```python
# Split training set into train/validation
from sklearn.model_selection import train_test_split
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

history = model.fit(
    X_train_fit, y_train_fit,
    validation_data=(X_val, y_val),  # ✅ PROPER VALIDATION
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=verbose
)

# Test set only used AFTER training
y_pred = model.predict(X_test)
```

---

## 🚨 CRITICAL ISSUE #2: INCOMPLETE EXPERIMENTS

**Current Status**:
- **Completed**: 8 experiments only
- **Required**: 330 experiments (300 binary + 30 multiclass)
- **Completion**: 2.4%

**Evidence**:
```bash
$ wc -l experiments/results/*.csv
   4 experiments/results/results_20260916_212126.csv
   9 experiments/results/results_20260916_213750.csv
  13 total
```

**Breakdown**:
- ✅ 4 models × 2 sizes × multiclass = 8 experiments completed
- ❌ 292 binary experiments missing
- ❌ 22 multiclass experiments missing
- ❌ MLP experiments: 0
- ❌ CNN experiments: 0

---

## 🚨 CRITICAL ISSUE #3: NO STABILITY ANALYSIS

**Required**: Multi-seed experiments (seeds: 42, 43, 44, 45, 46)

**Current**: Single seed (42) only

**Missing**:
- Mean accuracy across seeds
- Standard deviation across seeds
- Confidence intervals
- Stability metrics

---

## 🚨 CRITICAL ISSUE #4: NO CROSS-SIZE GENERALIZATION

**Required**: Train on size A, test on size B

**Current**: All experiments use same size for train/test

**Missing Examples**:
- Train 1KB → Test 8KB
- Train 1KB → Test 512KB
- Train 512KB → Test 1KB
- All cross-size combinations

---

## ⚠️ ISSUE #5: FRAMEWORK MISMATCH

**Synopsis Claims**: PyTorch for MLP/CNN

**Actual Implementation**: TensorFlow/Keras

**Files**:
```python
# src/models/mlp.py
import keras
from keras import layers, models, ops

# src/models/cnn.py
import keras
from keras import layers, models, ops
```

**Action Required**: Update synopsis/documentation to match implementation

---

## ⚠️ ISSUE #6: HKNNRF NOT FULLY VERIFIED

**Status**: Implementation exists but needs validation

**Verification Needed**:
1. Feature combination (original + RF-derived) - test passes
2. Binary classification - needs more experiments
3. Performance vs paper baseline
4. All 10 binary pairs tested

---

## 📊 EXPERIMENT STATUS MATRIX

### Multiclass (5-class)

| Size | SVM | KNN | RF | HKNNRF | MLP | CNN | Total |
|------|-----|-----|----|----|-----|-----|-------|
| 1KB | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0/6 |
| 8KB | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0/6 |
| 64KB | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0/6 |
| 256KB | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 4/6 |
| 512KB | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | 4/6 |
| **Total** | **2/5** | **2/5** | **2/5** | **2/5** | **0/5** | **0/5** | **8/30** |

### Binary Classification

| Pair | 1KB | 8KB | 64KB | 256KB | 512KB | Total |
|------|-----|-----|------|-------|-------|-------|
| AES & 3DES | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| AES & Blowfish | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| AES & CAST | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| AES & RC2 | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| 3DES & Blowfish | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| 3DES & CAST | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| 3DES & RC2 | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| Blowfish & CAST | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| Blowfish & RC2 | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| CAST & RC2 | ❌ | ❌ | ❌ | ❌ | ❌ | 0/5 |
| **Total** | **0/10** | **0/10** | **0/10** | **0/10** | **0/10** | **0/50** |

**Per model (each pair × 5 sizes)**:
- SVM: 0/50 binary experiments
- KNN: 0/50 binary experiments  
- RF: 0/50 binary experiments
- HKNNRF: 0/50 binary experiments
- MLP: 0/50 binary experiments
- CNN: 0/50 binary experiments

**Grand Total**: 0/300 binary experiments

---

## 📋 DATASET VERIFICATION

**Total Datasets**: 55 CSV files ✅

**Multiclass**: 5 files ✅
- 1KB.csv, 8KB.csv, 64KB.csv, 256KB.csv, 512KB.csv

**Binary**: 50 files ✅
- 10 pairs × 5 sizes = 50 files

**Data Quality**: Validated (no NaN, no inf, balanced classes) ✅

---

## 🔧 REQUIRED FIXES (Priority Order)

### URGENT (MUST FIX)

1. **Fix MLP/CNN Test Set Leakage**
   - Modify `src/models/mlp.py`
   - Modify `src/models/cnn.py`
   - Split training set into train/validation
   - Keep test set untouched until final evaluation
   - **Estimated Time**: 30 minutes
   - **Impact**: Makes MLP/CNN results valid

2. **Complete Primary Experiment Matrix**
   - Run remaining 22 multiclass experiments
   - Run all 300 binary experiments
   - Store results properly
   - **Estimated Time**: 4-6 hours
   - **Impact**: Required for project completion

3. **Update Documentation**
   - Fix PyTorch → TensorFlow/Keras
   - Update synopsis
   - Update README
   - **Estimated Time**: 15 minutes
   - **Impact**: Consistency

### IMPORTANT (SHOULD DO)

4. **Implement Stability Analysis**
   - Multi-seed runner (5 seeds)
   - Calculate mean/std metrics
   - Store stability results
   - **Estimated Time**: 1 hour
   - **Impact**: Required by synopsis

5. **Implement Cross-Size Generalization**
   - Cross-size experiment runner
   - Train/test on different sizes
   - Store cross-size results
   - **Estimated Time**: 1-2 hours
   - **Impact**: Required by synopsis

6. **Verify HKNNRF on Binary Tasks**
   - Run HKNNRF on all 50 binary experiments
   - Validate feature combination
   - Compare to paper results
   - **Estimated Time**: Included in #2
   - **Impact**: Paper replication validation

---

## 📈 COMPLETION ESTIMATE

### Current State
- **Implementation**: ~90% complete (code exists)
- **Testing**: ~2.4% complete (8/330 experiments)
- **Validation**: Incomplete (test leakage issue)
- **Documentation**: ~80% complete (needs framework fix)

### To Reach 100%
1. Fix test leakage: 30 minutes
2. Run 322 experiments: 4-6 hours
3. Stability analysis: 1 hour
4. Cross-size experiments: 1-2 hours
5. Documentation updates: 15 minutes
6. Final verification: 30 minutes

**Total Estimated Time**: 7-10 hours

---

## 🎯 IMMEDIATE ACTION PLAN

### Step 1: Fix Test Leakage (NOW)
- Modify MLP train function
- Modify CNN train function
- Test with small experiment

### Step 2: Run Primary Experiments (TODAY)
- Complete 22 multiclass experiments
- Run 300 binary experiments
- Verify results stored properly

### Step 3: Advanced Analysis (NEXT)
- Multi-seed stability
- Cross-size generalization

### Step 4: Finalize (LAST)
- Update documentation
- Clean repository
- Final verification

---

## ❌ FALSE CLAIMS TO CORRECT

Previous documentation claimed:
- ✗ "100% complete"
- ✗ "Production ready"
- ✗ "All experiments completed"
- ✗ "16/16 tests passed" (tests don't validate actual experiments)
- ✗ "PyTorch framework" (actually TensorFlow)

**Reality**:
- ✓ 2.4% experiments complete (8/330)
- ✓ Critical test leakage in MLP/CNN
- ✓ No stability analysis
- ✓ No cross-size experiments
- ✓ TensorFlow/Keras framework

---

**CONCLUSION**: Project has solid infrastructure but is **NOT COMPLETE**. 
Critical fixes needed before submission.

**Status**: 🔴 **INCOMPLETE - FIX REQUIRED**
