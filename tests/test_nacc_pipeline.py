"""End-to-end integration tests for the NACC framework."""

import numpy as np
import pytest
from src.nacc.datasets import generate_synthetic_noisy_data
from src.nacc.metrics import evaluate_external
from src.nacc.nacc import NoiseAwareConsensusClustering


def test_nacc_end_to_end_fit_predict():
    # Synthesize small high-dimensional cohort
    X, y_true, meta = generate_synthetic_noisy_data(
        n_samples=100,
        d_signal=4,
        d_noise=96,
        n_clusters=2,
        cluster_std=0.8,
        random_state=42,
    )

    model = NoiseAwareConsensusClustering(
        n_clusters=2,
        n_base_models=8,
        feature_optimizer="ga",
        max_optimization_iter=4,
        profiling_repeats=2,
        random_state=42,
    )

    pred = model.fit_predict(X)
    assert len(pred) == 100
    assert len(np.unique(pred)) == 2

    # High recovery despite 96% noise
    scores = evaluate_external(y_true, pred)
    assert scores["ari"] > 0.70

    # Verify model diagnostics
    summary = model.get_summary_dict()
    assert summary["core_percentage"] > 20.0
    assert summary["active_features_count"] > 0
    assert summary["mean_cluster_reliability"] > 0.0

    # Inductive test query projection
    X_test, _, _ = generate_synthetic_noisy_data(
        n_samples=20,
        d_signal=4,
        d_noise=96,
        n_clusters=2,
        cluster_std=0.8,
        random_state=142,
    )
    test_pred = model.predict(X_test)
    assert len(test_pred) == 20
