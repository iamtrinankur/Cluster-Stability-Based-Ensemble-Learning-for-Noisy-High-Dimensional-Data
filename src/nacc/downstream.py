"""Downstream Supervised Predictive Utility Validation.

Implements Section 9 of the NACC framework:
Evaluates whether discovered cluster structure provides useful separation
for supervised prediction using:
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Support Vector Machine (SVM)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def evaluate_downstream_utility(
    X: np.ndarray,
    y: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
    active_features: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluates KNN, Random Forest, and SVM classification performance.

    Can evaluate in either:
    1. Active feature subspace selected by NACC.
    2. Feature representation augmented by cluster affinity.

    Args:
        X: Feature matrix (N, D)
        y: Ground truth target labels (N,)
        cluster_labels: Optional cluster assignments to append as structural feature
        active_features: Optional subset of informative feature indices
        n_splits: Number of cross-validation folds
        random_state: Seed for cross-validation splitting

    Returns:
        Dictionary mapping model names ('knn', 'rf', 'svm') to accuracy and f1_macro.
    """
    # Filter out unlabeled (-1 or NaN) samples if any
    valid_mask = (y >= 0) & (~np.isnan(y))
    if np.sum(valid_mask) < 10:
        return {
            "knn": {"accuracy": 0.0, "f1_macro": 0.0},
            "rf": {"accuracy": 0.0, "f1_macro": 0.0},
            "svm": {"accuracy": 0.0, "f1_macro": 0.0},
        }

    X_val = X[valid_mask]
    y_val = y[valid_mask]

    # Select active features if specified
    if active_features is not None:
        X_val = X_val[:, active_features]

    # Optionally augment with cluster label one-hot
    if cluster_labels is not None:
        c_val = cluster_labels[valid_mask]
        n_clusters = len(np.unique(c_val))
        one_hot = np.zeros((len(c_val), n_clusters))
        for i, c in enumerate(c_val):
            if c >= 0 and c < n_clusters:
                one_hot[i, c] = 1.0
        X_val = np.hstack([X_val, one_hot])

    # Minimum samples per class check for StratifiedKFold
    _, class_counts = np.unique(y_val, return_counts=True)
    min_class_count = np.min(class_counts)
    actual_splits = max(2, min(n_splits, min_class_count))

    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    models = {
        "knn": KNeighborsClassifier(n_neighbors=min(5, max(1, min_class_count))),
        "rf": RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10),
        "svm": SVC(kernel="rbf", C=1.0, random_state=random_state),
    }

    results = {}
    for name, clf in models.items():
        try:
            acc_scores = cross_val_score(clf, X_val, y_val, cv=cv, scoring="accuracy")
            f1_scores = cross_val_score(clf, X_val, y_val, cv=cv, scoring="f1_macro")
            results[name] = {
                "accuracy": round(float(np.mean(acc_scores)), 4),
                "accuracy_std": round(float(np.std(acc_scores)), 4),
                "f1_macro": round(float(np.mean(f1_scores)), 4),
                "f1_macro_std": round(float(np.std(f1_scores)), 4),
            }
        except Exception:
            results[name] = {
                "accuracy": 0.0,
                "accuracy_std": 0.0,
                "f1_macro": 0.0,
                "f1_macro_std": 0.0,
            }

    return results
