"""Tests for theoretical spectral analysis and biological biomarker enrichment."""

import numpy as np
import pytest
from src.nacc.biology import evaluate_biomarker_enrichment
from src.nacc.datasets import load_dataset
from src.nacc.nacc import NoiseAwareConsensusClustering
from src.nacc.theory import compute_spectral_gap, verify_spectral_eigengap_expansion


def test_compute_spectral_gap():
    # Ideal 2-cluster block diagonal matrix
    A = np.zeros((20, 20))
    A[:10, :10] = 1.0
    A[10:, 10:] = 1.0

    gap, evals = compute_spectral_gap(A, k_clusters=2)
    assert gap > 0.5
    assert len(evals) == 20


def test_verify_spectral_eigengap_expansion():
    res = verify_spectral_eigengap_expansion(
        n_samples=60, k_clusters=3, noise_ratio=0.30, n_trials=5, random_state=42
    )
    assert res["expansion_confirmed"] is True
    assert res["eigengap_expansion_ratio"] > 1.0


def test_evaluate_biomarker_enrichment():
    feature_names = [f"GENE_{i}" for i in range(500)]
    feature_names[5] = "M27891_at"  # CST3 AML marker
    feature_names[10] = "M23197_at"  # CD33 AML marker
    feature_names[20] = "M28170_at"  # CD19 ALL marker

    selected_indices = np.array([5, 10, 20, 45, 99])
    res = evaluate_biomarker_enrichment(feature_names, selected_indices)

    assert res["overlap_count"] == 3
    assert res["fold_enrichment"] > 1.0
    assert res["is_statistically_significant"] is True
