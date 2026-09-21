"""
SVM + Naive Bayes soft-voting ensemble for CipherBench (project extension).

The base paper evaluates SVM, KNN, RF and HKNNRF. This model combines two
complementary learners on the same ten NIST p-value features:

- a margin-based RBF SVM on standardised features, and
- a Gaussian Naive Bayes model (generative, per-feature likelihoods).

Their class probabilities are averaged (soft voting). It is a plain
scikit-learn estimator (fit / predict / predict_proba / classes_), so it
plugs into the same protocol as the other classical models.
"""

from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_svmnb_model(C: float = 1.0, random_state: int = 0) -> VotingClassifier:
    """
    Create the SVM + Naive Bayes soft-voting classifier.

    Args:
        C: SVM regularisation strength
        random_state: seed for the SVM's probability calibration
    """
    svm = make_pipeline(
        StandardScaler(),
        SVC(C=C, kernel="rbf", probability=True, random_state=random_state),
    )
    return VotingClassifier(
        estimators=[("svm", svm), ("nb", GaussianNB())],
        voting="soft",
    )
