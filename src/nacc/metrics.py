"""Validation metrics for clustering quality, stability, and external correspondence.

Implements:
- Internal Validation:
  * Silhouette Score
  * Davies-Bouldin Index
  * Dunn Index
  * Calinski-Harabasz Index
- External Validation:
  * Adjusted Rand Index (ARI)
  * Normalized Mutual Information (NMI)
  * Adjusted Mutual Information (AMI)
  * Fowlkes-Mallows Index (FMI)
  * Clustering Purity
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def clustering_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes cluster purity.

    Purity measures the extent to which clusters contain a single class:
    Purity(Omega, C) = (1 / N) * sum_k max_j |omega_k cap c_j|
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    total_correct = 0
    for c in clusters:
        mask = (y_pred == c)
        if not np.any(mask):
            continue
        sub_labels = y_true[mask]
        counts = [np.sum(sub_labels == k) for k in classes]
        total_correct += max(counts)

    return float(total_correct / len(y_true))


def dunn_index(X: np.ndarray, labels: np.ndarray, max_samples: int = 1500) -> float:
    """Computes Dunn's Index of clustering separation to compactness.

    D = min_{i != j} d(C_i, C_j) / max_k diam(C_k)
    where d(C_i, C_j) is the inter-cluster distance and diam(C_k) is the intra-cluster diameter.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]  # ignore noise if -1

    if len(unique_labels) < 2:
        return 0.0

    # Subsample if dataset is large for computational efficiency
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_sub = X[idx]
        labels_sub = labels[idx]
    else:
        X_sub = X
        labels_sub = labels

    clusters = [X_sub[labels_sub == k] for k in unique_labels]
    # Filter empty clusters
    clusters = [c for c in clusters if len(c) > 0]
    if len(clusters) < 2:
        return 0.0

    # Maximum intra-cluster diameter
    max_diam = 0.0
    for c in clusters:
        if len(c) <= 1:
            continue
        dists = cdist(c, c, metric="euclidean")
        diam = np.max(dists)
        if diam > max_diam:
            max_diam = diam

    if max_diam == 0.0:
        max_diam = 1e-8

    # Minimum inter-cluster distance
    min_inter = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dists = cdist(clusters[i], clusters[j], metric="euclidean")
            min_d = np.min(dists)
            if min_d < min_inter:
                min_inter = min_d

    if np.isinf(min_inter):
        return 0.0

    return float(min_inter / max_diam)


def evaluate_internal(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Computes standard internal clustering validation indices."""
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {
            "silhouette": 0.0,
            "davies_bouldin": 99.0,
            "calinski_harabasz": 0.0,
            "dunn_index": 0.0,
        }

    try:
        # If dataset is very large, subsample for silhouette
        if len(labels) > 2000:
            sil = float(silhouette_score(X, labels, sample_size=2000, random_state=42))
        else:
            sil = float(silhouette_score(X, labels))
    except Exception:
        sil = 0.0

    try:
        db = float(davies_bouldin_score(X, labels))
    except Exception:
        db = 99.0

    try:
        ch = float(calinski_harabasz_score(X, labels))
    except Exception:
        ch = 0.0

    try:
        dunn = float(dunn_index(X, labels))
    except Exception:
        dunn = 0.0

    return {
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
        "dunn_index": round(dunn, 4),
    }


def evaluate_external(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard external validation metrics against reference ground-truth labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Filter out unassigned or NaN values in ground truth
    valid_mask = ~((y_true == -1) | (y_pred == -1) | (pd_isna(y_true)))
    if not np.any(valid_mask):
        return {"ari": 0.0, "nmi": 0.0, "ami": 0.0, "fmi": 0.0, "purity": 0.0}

    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    ari = float(adjusted_rand_score(y_t, y_p))
    nmi = float(normalized_mutual_info_score(y_t, y_p, average_method="arithmetic"))
    ami = float(adjusted_mutual_info_score(y_t, y_p, average_method="arithmetic"))
    fmi = float(fowlkes_mallows_score(y_t, y_p))
    purity = float(clustering_purity(y_t, y_p))

    return {
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "ami": round(ami, 4),
        "fmi": round(fmi, 4),
        "purity": round(purity, 4),
    }


def pd_isna(arr: np.ndarray) -> np.ndarray:
    """Helper to detect NaN values without requiring pandas import."""
    try:
        return np.isnan(arr.astype(float))
    except (ValueError, TypeError):
        return np.zeros(len(arr), dtype=bool)


def evaluate_all(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    X: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Unified evaluation returning both external and internal validation metrics."""
    results: Dict[str, float] = {}
    if y_true is not None:
        results.update(evaluate_external(y_true, y_pred))

    if X is not None:
        results.update(evaluate_internal(X, y_pred))

    return results
