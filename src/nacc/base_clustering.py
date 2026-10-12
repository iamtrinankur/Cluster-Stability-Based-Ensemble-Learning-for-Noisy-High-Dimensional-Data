"""Noise-Aware Base Clustering Generator.

Implements Section 4 of the NACC framework:
- Diverse view generation via Random Projections, DIVI-style feature gating,
  and controlled perturbations (noise injection, feature dropout, sample bagging).
- Multi-algorithm base partitioning (K-Means, GMM, Ward Linkage).
- Co-association matrix generation and base partition reliability weighting.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


@dataclass
class BasePartition:
    """Represents a single base clustering partition."""
    view_id: int
    labels: np.ndarray
    connectivity: np.ndarray  # (N, N) binary or sparse co-association
    reliability_weight: float  # w_m
    view_type: str  # 'random_projection', 'divi_gated', 'perturbed', etc.
    active_features: Optional[np.ndarray] = None


def compute_bimodality_coefficient(x: np.ndarray) -> float:
    """Computes Sarle's Bimodality Coefficient for an array:

    BC = (gamma^2 + 1) / (kappa + 3 * (n-1)^2 / ((n-2)*(n-3)))
    where gamma is skewness and kappa is kurtosis.
    Values > 5/9 (~0.555) indicate potential bimodality/multimodality.
    """
    n = len(x)
    if n < 4:
        return 0.0
    var = np.var(x)
    if var < 1e-12:
        return 0.0

    skew = float(stats.skew(x, bias=False))
    kurt_excess = float(stats.kurtosis(x, fisher=True, bias=False))

    denominator = kurt_excess + (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    if denominator <= 1e-4:
        denominator = 0.1
    bc = (skew ** 2 + 1.0) / denominator
    return float(np.clip(bc, 0.0, 1.0))


def divi_feature_gating(
    X: np.ndarray,
    top_fraction: float = 0.25,
    min_features: int = 10,
    max_features: int = 500,
) -> np.ndarray:
    """DIVI-style feature gating for noise suppression.

    Scores features using variance dispersion and bimodality coefficient.
    Unimodal background noise features are filtered out.

    Returns:
        Indices of selected informative features.
    """
    n_samples, n_features = X.shape
    if n_features <= min_features:
        return np.arange(n_features)

    # 1. Variance score
    variances = np.var(X, axis=0)

    # 2. Bimodality coefficients on candidate high-variance features
    candidate_size = min(n_features, max(max_features * 2, 200))
    top_var_idx = np.argsort(variances)[-candidate_size:]

    bc_scores = np.zeros(candidate_size)
    for i, idx in enumerate(top_var_idx):
        bc_scores[i] = compute_bimodality_coefficient(X[:, idx])

    # Combined ranking: normalized variance + bimodality score
    norm_var = (variances[top_var_idx] - np.min(variances[top_var_idx])) / (
        np.ptp(variances[top_var_idx]) + 1e-9
    )
    combined_scores = 0.5 * norm_var + 0.5 * bc_scores

    k_select = max(min_features, min(max_features, int(n_features * top_fraction)))
    selected_in_cand = np.argsort(combined_scores)[-k_select:]
    selected_indices = top_var_idx[selected_in_cand]

    return np.sort(selected_indices)


def create_connectivity_matrix(labels: np.ndarray) -> np.ndarray:
    """Computes N x N binary co-association matrix Am where Am(i, j) = 1 if labels[i] == labels[j]."""
    labels = np.asarray(labels)
    # Binary equivalence outer product
    return (labels[:, None] == labels[None, :]).astype(np.float64)


class NoiseAwareBaseClusterer:
    """Generates diverse, noise-aware base clusterings across multiple views and representations."""

    def __init__(
        self,
        n_clusters: int = 3,
        n_base_models: int = 20,
        subspace_ratio: float = 0.2,
        perturbation_noise_std: float = 0.05,
        subsample_ratio: float = 0.85,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.subspace_ratio = subspace_ratio
        self.perturbation_noise_std = perturbation_noise_std
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

    def generate_views(
        self,
        X: np.ndarray,
        feature_weights: Optional[np.ndarray] = None,
    ) -> List[BasePartition]:
        """Generates an ensemble of base partitions from diverse views.

        Args:
            X: Input matrix (N, D)
            feature_weights: Optional feature weights W from GA/PSO optimization (D,)

        Returns:
            List of BasePartition objects with labels, connectivity, and reliability weights.
        """
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)
        partitions: List[BasePartition] = []

        # If feature weights are provided, modulate X
        if feature_weights is not None:
            # Mask or scale features
            effective_X = X * feature_weights[None, :]
            # Select non-zero or top features for gating views
            active_feat_idx = np.where(feature_weights > np.median(feature_weights))[0]
            if len(active_feat_idx) < 5:
                active_feat_idx = np.argsort(feature_weights)[-min(D, 20):]
        else:
            effective_X = X
            active_feat_idx = np.arange(D)

        # Distribute views across diverse strategies:
        # 1. DIVI-style feature gated views (35%)
        # 2. Random projection views (35%)
        # 3. Controlled perturbation / bootstrap views (30%)
        n_divi = max(2, int(0.35 * self.n_base_models))
        n_proj = max(2, int(0.35 * self.n_base_models))
        n_pert = max(2, self.n_base_models - n_divi - n_proj)

        view_counter = 0

        # Strategy 1: DIVI Feature-Gated Views
        gated_features = divi_feature_gating(
            effective_X[:, active_feat_idx],
            top_fraction=self.subspace_ratio,
            min_features=max(3, min(D, 10)),
            max_features=min(D, 300),
        )
        base_gated_idx = active_feat_idx[gated_features]

        for i in range(n_divi):
            seed = int(rng.integers(1, 100000))
            # Subsample features from the gated pool with slight variation
            k_feats = max(3, int(len(base_gated_idx) * rng.uniform(0.7, 1.0)))
            view_feats = rng.choice(base_gated_idx, size=k_feats, replace=False)
            X_view = effective_X[:, view_feats]

            # Alternate clusterer
            if i % 3 == 0:
                clusterer = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=3)
            elif i % 3 == 1:
                clusterer = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
            else:
                clusterer = MiniBatchKMeans(
                    n_clusters=self.n_clusters, random_state=seed, batch_size=min(128, N), n_init=3
                )

            labels = clusterer.fit_predict(X_view)
            A_m = create_connectivity_matrix(labels)
            weight = self._estimate_partition_reliability(X_view, labels)

            partitions.append(
                BasePartition(
                    view_id=view_counter,
                    labels=labels,
                    connectivity=A_m,
                    reliability_weight=weight,
                    view_type="divi_gated",
                    active_features=view_feats,
                )
            )
            view_counter += 1

        # Strategy 2: Random Projection Views
        d_active = len(active_feat_idx)
        n_components = max(3, min(d_active, int(np.ceil(4 * np.log(N) / (0.3 ** 2)))))
        n_components = min(n_components, d_active, 48)

        for i in range(n_proj):
            seed = int(rng.integers(1, 100000))
            if i % 2 == 0:
                rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
            else:
                rp = SparseRandomProjection(n_components=n_components, random_state=seed)

            try:
                X_proj = rp.fit_transform(effective_X[:, active_feat_idx])
            except Exception:
                # Fallback to random feature subset if projection fails
                sub_feats = rng.choice(active_feat_idx, size=min(d_active, n_components), replace=False)
                X_proj = effective_X[:, sub_feats]

            clusterer = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=3)
            labels = clusterer.fit_predict(X_proj)
            A_m = create_connectivity_matrix(labels)
            weight = self._estimate_partition_reliability(X_proj, labels)

            partitions.append(
                BasePartition(
                    view_id=view_counter,
                    labels=labels,
                    connectivity=A_m,
                    reliability_weight=weight,
                    view_type="random_projection",
                    active_features=None,
                )
            )
            view_counter += 1

        # Strategy 3: Controlled Perturbation & Bootstrap Views
        for i in range(n_pert):
            seed = int(rng.integers(1, 100000))
            # Work within active feature subspace to avoid noise overwhelming
            X_active = effective_X[:, active_feat_idx]
            d_act = X_active.shape[1]
            noise = rng.normal(0.0, self.perturbation_noise_std, size=X_active.shape)
            X_pert = X_active + noise

            # Feature dropout within active features
            dropout_mask = rng.random(d_act) > 0.25
            if not np.any(dropout_mask):
                dropout_mask = np.ones(d_act, dtype=bool)
            X_pert = X_pert[:, dropout_mask]

            clusterer = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=3)
            labels = clusterer.fit_predict(X_pert)
            A_m = create_connectivity_matrix(labels)
            weight = self._estimate_partition_reliability(X_pert, labels)

            partitions.append(
                BasePartition(
                    view_id=view_counter,
                    labels=labels,
                    connectivity=A_m,
                    reliability_weight=weight,
                    view_type="perturbed",
                    active_features=active_feat_idx[dropout_mask],
                )
            )
            view_counter += 1

        # Normalize reliability weights to average 1.0
        weights = np.array([p.reliability_weight for p in partitions])
        if np.sum(weights) > 0:
            norm_weights = weights / (np.mean(weights) + 1e-9)
            for p, w in zip(partitions, norm_weights):
                p.reliability_weight = float(np.clip(w, 0.1, 3.0))

        return partitions

    def _estimate_partition_reliability(self, X_view: np.ndarray, labels: np.ndarray) -> float:
        """Estimates reliability weight wm based on partition balance and compactness."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2:
            return 0.1

        # 1. Entropy / Balance score (penalize degenerate partitions where one cluster has 99% of points)
        probs = counts / len(labels)
        entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(self.n_clusters)
        balance_weight = float(np.clip(entropy, 0.1, 1.0))

        # 2. Compactness score: mean intra-cluster dispersion vs total dispersion
        total_var = np.var(X_view)
        if total_var < 1e-9:
            return balance_weight

        intra_vars = []
        for k in unique_labels:
            mask = (labels == k)
            if np.sum(mask) > 1:
                intra_vars.append(np.var(X_view[mask]))
        avg_intra = np.mean(intra_vars) if intra_vars else total_var
        compactness = max(0.0, 1.0 - (avg_intra / (total_var + 1e-9)))

        raw_reliability = 0.5 * balance_weight + 0.5 * compactness
        return float(np.clip(raw_reliability, 0.2, 1.0))
