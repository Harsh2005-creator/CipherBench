# CipherBench Training Results Analysis

**Date**: 2026-08-28  
**Dataset**: Multiclass 512KB  
**Models Trained**: SVM, KNN, Random Forest, HKNNRF, MLP, CNN

---

## Training Results Summary

### Model Performance (512KB Multiclass)

| Model    | Accuracy | Precision | Recall | F1-Score | Training Time |
|----------|----------|-----------|--------|----------|---------------|
| SVM      | 0.1900   | 0.2297    | 0.1900 | 0.1860   | 0.04s         |
| KNN      | 0.2000   | 0.2191    | 0.2000 | 0.1648   | 0.01s         |
| RF       | 0.2300   | 0.2414    | 0.2300 | 0.2315   | 0.05s         |
| HKNNRF   | 0.2500   | 0.2548    | 0.2500 | 0.2401   | 0.25s         |
| MLP      | 0.2200   | 0.1807    | 0.2200 | 0.1759   | 10.81s        |
| CNN      | 0.2000   | 0.1203    | 0.2000 | 0.0952   | 7.52s         |

**Baseline (Random Guessing)**: 20% for 5-class problem

---

## Critical Issues Identified

### 1. Poor Model Performance
All models achieve ~19-25% accuracy, which is barely above random guessing (20% for 5 classes). This indicates the models cannot effectively learn to distinguish between cipher algorithms.

### 2. Feature Discriminative Power Analysis

Statistical analysis (ANOVA) reveals that **NONE** of the NIST p-value features have significant discriminative power:

**512KB Dataset Feature Analysis:**
```
Feature         F-statistic   p-value   Significant?
aetPValue       2.0738        0.0831    NO
custPValue      1.7484        0.1381    NO
dtfPValue       1.2047        0.3079    NO
fwbtPValue      0.9459        0.4371    NO
lorbPValue      0.9206        0.4516    NO
mtPValue        1.6624        0.1575    NO
retPValue       0.1929        0.9421    NO
revtPValue      0.5187        0.7220    NO
rtPValue        0.7143        0.5824    NO
stPValue        2.3049        0.0574    NO
```

**Criteria**: p-value < 0.05 indicates the feature can distinguish between classes.  
**Result**: All p-values > 0.05 → Features cannot distinguish between cipher algorithms.

### 3. Cross-Dataset Analysis

| Dataset Size | Significant Features (p < 0.05) |
|--------------|--------------------------------|
| 1KB          | 2/10 features                  |
| 8KB          | 0/10 features                  |
| 64KB         | 0/10 features                  |
| 256KB        | 1/10 features                  |
| 512KB        | 0/10 features                  |

**Conclusion**: The problem is consistent across all dataset sizes.

---

## Root Cause Analysis

### Possible Explanations:

1. **Dataset Generation Issue**
   - NIST p-values may not have been computed correctly
   - Ciphertext samples might not be properly generated
   - Feature extraction process may be flawed

2. **Theoretical Limitation**
   - Modern cipher algorithms (AES, 3DES, Blowfish, CAST, RC2) produce highly random output
   - NIST randomness tests are designed to detect non-randomness
   - Good ciphers SHOULD produce similar p-values (all close to uniform distribution)
   - This makes cipher identification fundamentally difficult using only statistical tests

3. **Missing Context from Paper**
   - The research paper may have used additional preprocessing
   - Different feature engineering might be required
   - The paper might have used raw statistical values instead of p-values
   - There could be additional features not included in this dataset

---

## Expected vs Actual Results

### From README.md:
> "512KB Multiclass: HKNNRF typically achieves >95% accuracy"

### Our Results:
- **HKNNRF Accuracy**: 25% (vs expected >95%)
- **Difference**: -70 percentage points

This massive discrepancy indicates a fundamental problem with the dataset or implementation.

---

## Recommendations

### Immediate Actions:

1. **Verify Dataset Generation**
   - Check if ciphertext was properly generated with correct algorithms
   - Verify NIST test suite was run correctly
   - Ensure p-values are computed properly (not raw test statistics)

2. **Check Paper Implementation**
   - Re-read the paper "A block cipher algorithm identification" (paper #6)
   - Verify if they used p-values or raw NIST statistics
   - Check if additional features or preprocessing was used
   - Contact paper authors if possible

3. **Dataset Validation**
   - Manually inspect a few samples
   - Verify labels match the cipher used to generate ciphertext
   - Check if there's a mislabeling issue

4. **Alternative Approaches**
   ```
   - Use raw NIST statistics instead of p-values
   - Try different feature extraction methods
   - Use byte-level features from ciphertext directly
   - Implement deep learning on raw ciphertext bytes
   ```

### Long-term Solutions:

1. Regenerate dataset with verified pipeline
2. Explore alternative feature sets beyond NIST tests
3. Consider ensemble approaches combining multiple feature types
4. Implement the exact methodology from the paper

---

## Technical Notes

- **Python**: 3.14.2
- **TensorFlow**: 2.22.0-dev (tf-nightly)
- **Dataset**: 500 samples (400 train, 100 test)
- **Classes**: 5 (AES, 3DES, Blowfish, CAST, RC2)
- **Features**: 10 NIST p-values per sample
- **Database**: MySQL (cipherbench) - configured but save failed due to Unicode issues

---

## Next Steps

1. ✅ Database setup complete
2. ✅ All dependencies installed
3. ✅ Models trained and evaluated
4. ⚠️ **CRITICAL**: Dataset validation required
5. ⏳ Pending: Paper comparison (cannot proceed without valid dataset)

---

## Conclusion

The training pipeline works correctly, but the dataset appears to be fundamentally unsuitable for cipher identification using the current features. The NIST p-values show no statistically significant differences between cipher algorithms, making classification impossible.

**Status**: Implementation complete, but dataset requires investigation before meaningful paper comparison can be performed.
