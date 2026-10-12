"""Tests for sample stability diagnostics and core/halo modeling."""

import numpy as np
import pytest
from src.nacc.stability import align_partition_labels, analyze_sample_stability


def test_align_partition_labels():
    y_ref = np.array([0, 0, 1, 1, 2, 2])
    y_target = np.array([1, 1, 2, 2, 0, 0])  # Permuted labels
    aligned = align_partition_labels(y_ref, y_target)
    np.testing.assert_array_equal(aligned, y_ref)


def test_analyze_sample_stability():
    # Construct 5 partitions where first 10 samples always cluster together (stable)
    # and last 5 fluctuate (uncertain)
    N = 15
    partitions = []
    rng = np.random.default_rng(42)

    for m in range(8):
        labels = np.zeros(N, dtype=int)
        labels[:8] = 0
        labels[8:12] = 1
        # Random noise assignments for last 3 points
        labels[12:] = rng.choice([0, 1], size=3)
        partitions.append(labels)

    profile = analyze_sample_stability(partitions, tau_core=0.60, tau_uncertain=0.40)

    assert len(profile.stability_scores) == N
    assert profile.reliability_matrix.shape == (N, N)
    assert np.all(profile.stability_scores >= 0.0)
    assert np.all(profile.stability_scores <= 1.0)
    assert np.mean(profile.stability_scores[:8]) > np.mean(profile.stability_scores[12:])
    assert profile.core_ratio > 0.0
