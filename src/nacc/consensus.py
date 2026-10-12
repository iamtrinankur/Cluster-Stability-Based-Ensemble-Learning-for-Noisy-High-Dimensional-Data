"""Error-Aware Consensus Fusion (Degree-Preserving Formulation).

Implements Section 7 of the NACC framework (Refactored for Spectral Graph Connectivity):
- Estimates base partition reliability w_m via consensus correlation.
- Fuses sample-level reliability R_m preserving strictly positive graph degrees (lambda_2 > 0).
- Integrates persistent dissimilarity via soft contrast regularization without zero-degree clipping.
- Performs normalized spectral graph partitioning with core-guided centroid calibration.
- Provides closed-form inductive out-of-sample projection for unseen test instances.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min

from src.nacc.base_clustering import BasePartition
from src.nacc.stability import SampleStabilityProfile


class ErrorAwareConsensus:
    """Fuses diverse base partitions into a robust consensus clustering."""

    def __init__(
        self,
        n_clusters: int = 3,
        dissimilarity_alpha: float = 0.20,
        reliability_blend_gamma: float = 0.60,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.dissimilarity_alpha = dissimilarity_alpha
        self.reliability_blend_gamma = reliability_blend_gamma
        self.random_state = random_state

        self.consensus_matrix_: Optional[np.ndarray] = None
        self.dissimilarity_matrix_: Optional[np.ndarray] = None
        self.final_labels_: Optional[np.ndarray] = None
        self.cluster_centroids_: Optional[np.ndarray] = None
        self.active_features_: Optional[np.ndarray] = None

    def fuse(
        self,
        partitions: List[BasePartition],
        stability_profile: SampleStabilityProfile,
        X: Optional[np.ndarray] = None,
        active_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes degree-preserving error-aware consensus matrix H and extracts cluster labels.

        Args:
            partitions: List of BasePartition objects
            stability_profile: SampleStabilityProfile containing R_m and sample weights
            X: Original or feature-weighted data matrix (N, D) for centroid computation
            active_features: Indices of active features used in projection

        Returns:
            final_labels: Integer cluster labels (N,)
        """
        M = len(partitions)
        N = len(partitions[0].labels)
        self.active_features_ = active_features

        # 1. Base partition reliability weighting
        # Mean unweighted co-association
        raw_coassoc = np.zeros((N, N), dtype=np.float64)
        for p in partitions:
            raw_coassoc += p.connectivity
        raw_coassoc /= M

        weights = np.zeros(M, dtype=np.float64)
        for m, p in enumerate(partitions):
            # Correlation between individual partition connectivity and ensemble average
            corr = float(np.corrcoef(p.connectivity.ravel(), raw_coassoc.ravel())[0, 1])
            agreement = max(0.1, corr if not np.isnan(corr) else 0.5)
            weights[m] = p.reliability_weight * (agreement ** 1.5)

        total_weight = float(np.sum(weights))
        if total_weight <= 0:
            weights = np.ones(M)
            total_weight = float(M)

        # 2. Weighted ensemble co-association matrix A_bar
        A_bar = np.zeros((N, N), dtype=np.float64)
        for m, p in enumerate(partitions):
            A_bar += weights[m] * p.connectivity
        A_bar /= total_weight
        np.fill_diagonal(A_bar, 1.0)

        # 3. Persistent separation matrix
        D_dissim = 1.0 - A_bar
        np.fill_diagonal(D_dissim, 0.0)

        # 4. Degree-Preserving Sample Reliability Attenuation:
        # H_rel = (1 - gamma) * A_bar + gamma * (R_m * A_bar)
        # Guarantees every node retains at least (1 - gamma)*A_bar connectivity (lambda_2 > 0)
        R_m = stability_profile.reliability_matrix
        gamma = self.reliability_blend_gamma
        H_rel = (1.0 - gamma) * A_bar + gamma * (R_m * A_bar)

        # 5. Soft Dissimilarity Contrast Regularization:
        # Instead of destructive elementwise subtraction that creates zero-degree isolates,
        # apply smooth quadratic contrast enhancement:
        # H_eff(i, j) = H_rel(i, j) * (1 - alpha * D_dissim(i, j)^2)
        alpha = np.clip(self.dissimilarity_alpha, 0.0, 0.5)
        H_eff = H_rel * (1.0 - alpha * (D_dissim ** 2))

        # Ensure diagonal is exactly 1.0 and matrix is symmetric
        np.fill_diagonal(H_eff, 1.0)
        H_eff = 0.5 * (H_eff + H_eff.T)
        H_eff = np.clip(H_eff, 1e-6, 1.0)  # Strictly positive minimum edge floor

        self.consensus_matrix_ = H_eff
        self.dissimilarity_matrix_ = D_dissim

        # 6. Spectral Partitioning on Connected Affinity Graph
        try:
            sc = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="precomputed",
                random_state=self.random_state,
                assign_labels="kmeans",
                n_init=10,
            )
            final_labels = sc.fit_predict(H_eff)
        except Exception:
            # Fallback to Agglomerative on precomputed dissimilarity
            from sklearn.cluster import AgglomerativeClustering
            dist_mat = 1.0 - H_eff
            np.fill_diagonal(dist_mat, 0.0)
            agg = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric="precomputed",
                linkage="average",
            )
            final_labels = agg.fit_predict(dist_mat)

        # Ensure contiguous labels 0..K-1
        unique_lbls = np.unique(final_labels)
        lbl_map = {lbl: i for i, lbl in enumerate(unique_lbls)}
        final_labels = np.array([lbl_map[l] for l in final_labels])

        self.final_labels_ = final_labels

        # 7. Core-Guided Centroid Calibration
        if X is not None:
            self._calibrate_centroids(X, final_labels, stability_profile.core_mask)

        return final_labels

    def _calibrate_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        core_mask: np.ndarray,
    ):
        """Calibrates cluster centroids using certified Core samples on active features."""
        k = self.n_clusters
        d = X.shape[1]
        centroids = np.zeros((k, d), dtype=np.float64)

        for c in range(k):
            c_mask = (labels == c)
            core_c = c_mask & core_mask
            if np.any(core_c):
                centroids[c] = np.mean(X[core_c], axis=0)
            elif np.any(c_mask):
                centroids[c] = np.mean(X[c_mask], axis=0)
            else:
                centroids[c] = np.mean(X, axis=0)

        self.cluster_centroids_ = centroids

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """Inductive out-of-sample projection for unseen query instances."""
        if self.cluster_centroids_ is None:
            raise ValueError("Model has not been fitted with feature matrix X yet.")

        if self.active_features_ is not None:
            feats = self.active_features_
            query_sub = X_query[:, feats]
            centroids_sub = self.cluster_centroids_[:, feats]
        else:
            query_sub = X_query
            centroids_sub = self.cluster_centroids_

        pred_labels, _ = pairwise_distances_argmin_min(query_sub, centroids_sub)
        return pred_labels
