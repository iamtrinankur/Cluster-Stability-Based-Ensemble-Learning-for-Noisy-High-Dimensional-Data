"""Ensemble Consensus Clustering Baselines.

Implements standard ensemble methodologies:
- Standard Consensus Clustering (Fred & Jain 2005)
- Core-Halo Consensus (Li et al. 2019)
"""

from typing import Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering


class StandardConsensusClustering:
    """Standard Consensus Clustering (Fred & Jain 2005).

    Constructs a uniform average co-association matrix from diverse K-Means runs,
    followed by agglomerative or spectral graph extraction.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_base_models: int = 20,
        subsample_ratio: float = 0.8,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state
        self.coassociation_matrix_: Optional[np.ndarray] = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)

        coassoc = np.zeros((N, N), dtype=np.float64)
        counts = np.zeros((N, N), dtype=np.float64)

        for m in range(self.n_base_models):
            seed = self.random_state + m * 31
            sub_n = max(int(self.subsample_ratio * N), self.n_clusters * 2)
            sub_idx = rng.choice(N, size=sub_n, replace=False)

            km = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=2)
            sub_labels = km.fit_predict(X[sub_idx])

            # Update co-association counts for subsampled pairs
            for i_idx, i in enumerate(sub_idx):
                for j_idx, j in enumerate(sub_idx):
                    counts[i, j] += 1.0
                    if sub_labels[i_idx] == sub_labels[j_idx]:
                        coassoc[i, j] += 1.0

        # Uniform average co-association
        coassoc = np.divide(coassoc, counts, out=np.zeros_like(coassoc), where=counts > 0)
        np.fill_diagonal(coassoc, 1.0)
        self.coassociation_matrix_ = coassoc

        # Extract final clusters via Average Linkage on dissimilarity
        dist_mat = 1.0 - coassoc
        np.fill_diagonal(dist_mat, 0.0)

        agg = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="precomputed",
            linkage="average",
        )
        return agg.fit_predict(dist_mat)


class CoreHaloConsensusClustering:
    """Core-Halo Consensus Clustering (Li et al. 2019).

    Identifies unambiguous core samples using a determinacy threshold,
    clusters the core graph, and projects halo instances to nearest core centroids.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_base_models: int = 20,
        core_threshold: float = 0.6,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.core_threshold = core_threshold
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)

        # 1. Base partitions
        partitions = []
        for m in range(self.n_base_models):
            seed = self.random_state + m * 47
            km = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=2)
            partitions.append(km.fit_predict(X))

        # 2. Co-association matrix
        coassoc = np.zeros((N, N), dtype=np.float64)
        for labels in partitions:
            coassoc += (labels[:, None] == labels[None, :]).astype(np.float64)
        coassoc /= self.n_base_models

        # 3. Determinacy-based core extraction
        # Determinacy score: mean of highest 25% co-associations minus mean of lowest 25%
        determinacy = np.zeros(N)
        for i in range(N):
            row = np.sort(coassoc[i])
            determinacy[i] = np.mean(row[-max(2, int(0.25 * N)):]) - np.mean(row[:max(2, int(0.25 * N))])

        core_mask = determinacy >= self.core_threshold
        if np.sum(core_mask) < self.n_clusters * 2:
            # Fallback to top 50%
            core_mask = determinacy >= np.median(determinacy)

        # 4. Cluster Core samples
        core_idx = np.where(core_mask)[0]
        core_dist = 1.0 - coassoc[np.ix_(core_idx, core_idx)]
        np.fill_diagonal(core_dist, 0.0)

        agg = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="precomputed",
            linkage="average",
        )
        core_labels = agg.fit_predict(core_dist)

        # 5. Assign Halo and Uncertain instances to nearest core cluster
        final_labels = np.zeros(N, dtype=int)
        final_labels[core_idx] = core_labels

        halo_idx = np.where(~core_mask)[0]
        for h in halo_idx:
            # Calculate affinity to each core cluster
            cluster_affinities = np.zeros(self.n_clusters)
            for k in range(self.n_clusters):
                k_core = core_idx[core_labels == k]
                if len(k_core) > 0:
                    cluster_affinities[k] = np.mean(coassoc[h, k_core])
            final_labels[h] = int(np.argmax(cluster_affinities))

        return final_labels
