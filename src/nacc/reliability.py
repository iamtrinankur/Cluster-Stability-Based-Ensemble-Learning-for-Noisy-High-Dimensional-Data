"""Cluster Reliability Profile and Fingerprinting.

Implements Section 8 of the NACC framework:
Evaluates cluster stability across 5 controlled perturbation dimensions:
1. Additive Noise
2. Feature Dropout
3. Sample Variation (Subsampling/Bagging)
4. Distribution Shift
5. Batch-like Effects

Constructs a multi-dimensional reliability fingerprint for each cluster.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class ClusterFingerprint:
    """Fingerprint of a single cluster's resilience across perturbation regimes."""
    cluster_id: int
    size: int
    noise_stability: float
    dropout_stability: float
    sample_stability: float
    shift_stability: float
    batch_stability: float
    overall_reliability: float
    stability_grade: str  # 'Rock-Solid' (>=0.75), 'Moderate' (0.5-0.75), 'Fragile' (<0.5)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.noise_stability,
            self.dropout_stability,
            self.sample_stability,
            self.shift_stability,
            self.batch_stability,
        ])


def match_clusters_jaccard(y_true: np.ndarray, y_pert: np.ndarray, n_clusters: int) -> np.ndarray:
    """Matches perturbed clusters to reference clusters and returns per-cluster Jaccard scores."""
    cost = np.zeros((n_clusters, n_clusters))
    jaccards = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        mask_true = (y_true == i)
        size_true = np.sum(mask_true)
        for j in range(n_clusters):
            mask_pert = (y_pert == j)
            intersection = np.sum(mask_true & mask_pert)
            union = np.sum(mask_true | mask_pert)
            jaccard = float(intersection / union) if union > 0 else 0.0
            jaccards[i, j] = jaccard
            cost[i, j] = -jaccard

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_scores = np.zeros(n_clusters)
    for r, c in zip(row_ind, col_ind):
        matched_scores[r] = jaccards[r, c]

    return matched_scores


class ClusterReliabilityProfiler:
    """Profiles cluster resilience across multiple controlled data perturbations."""

    def __init__(
        self,
        n_repeats: int = 5,
        noise_std: float = 0.15,
        dropout_rate: float = 0.25,
        subsample_ratio: float = 0.80,
        random_state: int = 42,
    ):
        self.n_repeats = n_repeats
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

    def profile(
        self,
        X: np.ndarray,
        reference_labels: np.ndarray,
        cluster_fn: Callable[[np.ndarray], np.ndarray],
    ) -> List[ClusterFingerprint]:
        """Profiles reference clusters across 5 perturbation regimes.

        Args:
            X: Original data matrix (N, D)
            reference_labels: Cluster labels to profile (N,)
            cluster_fn: Function that clusters an arbitrary (N', D') matrix and returns labels

        Returns:
            List of ClusterFingerprint objects, one per cluster.
        """
        N, D = X.shape
        unique_clusters = np.sort(np.unique(reference_labels))
        K = len(unique_clusters)
        rng = np.random.default_rng(self.random_state)

        # Storage for scores across repeats (n_repeats, K)
        noise_scores = np.zeros((self.n_repeats, K))
        dropout_scores = np.zeros((self.n_repeats, K))
        sample_scores = np.zeros((self.n_repeats, K))
        shift_scores = np.zeros((self.n_repeats, K))
        batch_scores = np.zeros((self.n_repeats, K))

        for rep in range(self.n_repeats):
            seed = self.random_state + rep * 17

            # 1. Additive Noise
            noise = rng.normal(0.0, self.noise_std, size=X.shape)
            X_noisy = X + noise
            try:
                y_noisy = cluster_fn(X_noisy)
                noise_scores[rep] = match_clusters_jaccard(reference_labels, y_noisy, K)
            except Exception:
                noise_scores[rep] = 0.5

            # 2. Feature Dropout
            dropout_mask = rng.random(D) > self.dropout_rate
            if not np.any(dropout_mask):
                dropout_mask = np.ones(D, dtype=bool)
            X_drop = X[:, dropout_mask]
            try:
                y_drop = cluster_fn(X_drop)
                dropout_scores[rep] = match_clusters_jaccard(reference_labels, y_drop, K)
            except Exception:
                dropout_scores[rep] = 0.5

            # 3. Sample Variation (Subsampling)
            sub_n = int(self.subsample_ratio * N)
            sub_idx = rng.choice(N, size=sub_n, replace=False)
            try:
                y_sub = cluster_fn(X[sub_idx])
                sample_scores[rep] = match_clusters_jaccard(reference_labels[sub_idx], y_sub, K)
            except Exception:
                sample_scores[rep] = 0.5

            # 4. Distribution Shift (Covariate Shift)
            shift_scale = rng.uniform(0.8, 1.2, size=(1, D))
            shift_offset = rng.normal(0.0, 0.1, size=(1, D))
            X_shift = (X * shift_scale) + shift_offset
            try:
                y_shift = cluster_fn(X_shift)
                shift_scores[rep] = match_clusters_jaccard(reference_labels, y_shift, K)
            except Exception:
                shift_scores[rep] = 0.5

            # 5. Batch-like Effects (Synthetic 2-batch shift)
            batch_split = rng.random(N) > 0.5
            batch_features = rng.choice(D, size=min(D, max(5, int(D * 0.15))), replace=False)
            X_batch = X.copy()
            X_batch[batch_split, :][:, batch_features] += 0.5
            try:
                y_batch = cluster_fn(X_batch)
                batch_scores[rep] = match_clusters_jaccard(reference_labels, y_batch, K)
            except Exception:
                batch_scores[rep] = 0.5

        # Aggregate mean scores per cluster
        fingerprints = []
        for k in range(K):
            s_noise = float(np.mean(noise_scores[:, k]))
            s_drop = float(np.mean(dropout_scores[:, k]))
            s_sample = float(np.mean(sample_scores[:, k]))
            s_shift = float(np.mean(shift_scores[:, k]))
            s_batch = float(np.mean(batch_scores[:, k]))

            overall = float(np.mean([s_noise, s_drop, s_sample, s_shift, s_batch]))
            size = int(np.sum(reference_labels == k))

            if overall >= 0.75:
                grade = "Rock-Solid"
            elif overall >= 0.50:
                grade = "Moderate"
            else:
                grade = "Fragile"

            fingerprints.append(
                ClusterFingerprint(
                    cluster_id=k,
                    size=size,
                    noise_stability=round(s_noise, 4),
                    dropout_stability=round(s_drop, 4),
                    sample_stability=round(s_sample, 4),
                    shift_stability=round(s_shift, 4),
                    batch_stability=round(s_batch, 4),
                    overall_reliability=round(overall, 4),
                    stability_grade=grade,
                )
            )

        return fingerprints
