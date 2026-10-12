"""Tests for base clustering generator and DIVI feature gating."""

import numpy as np
import pytest
from src.nacc.base_clustering import (
    NoiseAwareBaseClusterer,
    compute_bimodality_coefficient,
    create_connectivity_matrix,
    divi_feature_gating,
)


def test_bimodality_coefficient():
    rng = np.random.default_rng(42)
    # Unimodal normal distribution -> low BC (< 0.555)
    unimodal = rng.normal(0, 1, 500)
    bc_uni = compute_bimodality_coefficient(unimodal)
    assert bc_uni < 0.55

    # Bimodal mixture of two distant peaks -> higher BC
    bimodal = np.concatenate([rng.normal(-3, 0.5, 250), rng.normal(3, 0.5, 250)])
    bc_bi = compute_bimodality_coefficient(bimodal)
    assert bc_bi > bc_uni


def test_divi_feature_gating():
    rng = np.random.default_rng(42)
    # Signal features (first 5)
    signal = np.vstack([
        rng.normal(-3, 0.5, size=(50, 5)),
        rng.normal(3, 0.5, size=(50, 5)),
    ])
    # Noise features (next 45)
    noise = rng.normal(0, 1, size=(100, 45))
    X = np.hstack([signal, noise])

    selected = divi_feature_gating(X, top_fraction=0.2, min_features=5, max_features=15)
    assert len(selected) >= 5
    # True signal features should be recovered in the top pool
    recovered_signals = [i for i in range(5) if i in selected]
    assert len(recovered_signals) >= 3


def test_noise_aware_base_clusterer():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(60, 20))
    generator = NoiseAwareBaseClusterer(n_clusters=2, n_base_models=8, random_state=42)
    partitions = generator.generate_views(X)

    assert len(partitions) == 8
    for p in partitions:
        assert p.connectivity.shape == (60, 60)
        assert len(p.labels) == 60
        assert p.reliability_weight > 0.0
