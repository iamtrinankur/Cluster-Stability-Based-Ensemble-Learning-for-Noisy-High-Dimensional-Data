"""Tests for validation metrics."""

import numpy as np
import pytest
from src.nacc.metrics import clustering_purity, dunn_index, evaluate_all, evaluate_external, evaluate_internal


def test_perfect_clustering_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    ext = evaluate_external(y_true, y_pred)
    assert ext["ari"] == 1.0
    assert ext["nmi"] == 1.0
    assert ext["purity"] == 1.0


def test_permuted_clustering_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])  # Relabeled

    ext = evaluate_external(y_true, y_pred)
    assert ext["ari"] == 1.0
    assert ext["nmi"] == 1.0
    assert ext["purity"] == 1.0


def test_dunn_index_well_separated():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ])
    labels = np.array([0, 0, 1, 1])
    dunn = dunn_index(X, labels)
    assert dunn > 1.0


def test_evaluate_internal():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(0.0, 0.5, size=(30, 4)),
        rng.normal(5.0, 0.5, size=(30, 4)),
    ])
    labels = np.array([0] * 30 + [1] * 30)

    res = evaluate_internal(X, labels)
    assert "silhouette" in res
    assert res["silhouette"] > 0.5
    assert res["davies_bouldin"] < 1.0


def test_evaluate_all():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(0.0, 0.5, size=(20, 3)),
        rng.normal(5.0, 0.5, size=(20, 3)),
    ])
    y_true = np.array([0] * 20 + [1] * 20)
    y_pred = y_true.copy()

    all_scores = evaluate_all(y_true, y_pred, X)
    assert "ari" in all_scores
    assert "silhouette" in all_scores
