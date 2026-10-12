"""Tests for error-aware consensus fusion and inductive projection."""

import numpy as np
import pytest
from src.nacc.base_clustering import BasePartition
from src.nacc.consensus import ErrorAwareConsensus
from src.nacc.stability import analyze_sample_stability


def test_error_aware_consensus_fusion():
    N = 30
    K = 2
    rng = np.random.default_rng(42)

    # Synthetic data matrix
    X = np.vstack([
        rng.normal(-3, 0.5, size=(15, 4)),
        rng.normal(3, 0.5, size=(15, 4)),
    ])

    partitions = []
    for m in range(5):
        lbls = np.array([0] * 15 + [1] * 15)
        # Flip 1-2 labels to introduce slight noise
        flip_idx = rng.choice(N, size=2, replace=False)
        lbls[flip_idx] = 1 - lbls[flip_idx]
        conn = (lbls[:, None] == lbls[None, :]).astype(float)
        partitions.append(
            BasePartition(
                view_id=m,
                labels=lbls,
                connectivity=conn,
                reliability_weight=1.0,
                view_type="test",
            )
        )

    stability = analyze_sample_stability([p.labels for p in partitions])
    engine = ErrorAwareConsensus(n_clusters=K, random_state=42)
    final_labels = engine.fuse(partitions, stability, X=X, active_features=np.arange(4))

    assert len(final_labels) == N
    assert len(np.unique(final_labels)) == K
    assert engine.consensus_matrix_.shape == (N, N)

    # Inductive projection on new unseen query points
    X_query = np.array([[-3.0, -3.0, -3.0, -3.0], [3.0, 3.0, 3.0, 3.0]])
    query_preds = engine.predict(X_query)
    assert len(query_preds) == 2
    assert query_preds[0] != query_preds[1]  # Should map to opposite clusters
