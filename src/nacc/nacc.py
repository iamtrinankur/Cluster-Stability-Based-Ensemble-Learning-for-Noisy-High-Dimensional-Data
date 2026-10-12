"""Noise-Aware Consensus Clustering (NACC) Main Framework.

Unifies all stages from the NACC specification:
- Data Preprocessing
- GA/PSO-based Feature Optimization
- Noise-Aware Base Clustering
- Sample Stability and Core/Halo/Uncertain Modeling
- Error-Aware Consensus Fusion
- Cluster Reliability Profiling & Fingerprints
- Inductive Out-of-Sample Query Prediction
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from src.nacc.base_clustering import BasePartition, NoiseAwareBaseClusterer
from src.nacc.consensus import ErrorAwareConsensus
from src.nacc.optimizers import optimize_feature_weights
from src.nacc.reliability import ClusterFingerprint, ClusterReliabilityProfiler
from src.nacc.stability import SampleStabilityProfile, analyze_sample_stability


class NoiseAwareConsensusClustering(BaseEstimator, ClusterMixin):
    """Noise-Aware Consensus Clustering (NACC) estimator.

    Parameters:
        n_clusters: Target number of clusters (K)
        n_base_models: Number of base clustering views in the ensemble (M)
        feature_optimizer: 'ga' (Genetic Algorithm), 'pso' (Particle Swarm), or None
        subspace_ratio: Fraction of features retained in DIVI-style gating views
        tau_core: Threshold for certified Core sample identification
        tau_uncertain: Threshold for Uncertain sample identification
        dissimilarity_alpha: Weight of persistent dissimilarity regularization
        max_optimization_iter: Iterations for GA/PSO feature optimization
        profiling_repeats: Repeats for computing cluster reliability fingerprints
        random_state: Seed for reproducibility
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_base_models: int = 20,
        feature_optimizer: Optional[str] = "ga",
        subspace_ratio: float = 0.25,
        tau_core: float = 0.65,
        tau_uncertain: float = 0.35,
        dissimilarity_alpha: float = 0.15,
        max_optimization_iter: int = 12,
        profiling_repeats: int = 5,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.n_base_models = n_base_models
        self.feature_optimizer = feature_optimizer
        self.subspace_ratio = subspace_ratio
        self.tau_core = tau_core
        self.tau_uncertain = tau_uncertain
        self.dissimilarity_alpha = dissimilarity_alpha
        self.max_optimization_iter = max_optimization_iter
        self.profiling_repeats = profiling_repeats
        self.random_state = random_state

        # Fitted attributes
        self.labels_: Optional[np.ndarray] = None
        self.feature_weights_: Optional[np.ndarray] = None
        self.selected_features_: Optional[np.ndarray] = None
        self.base_partitions_: Optional[List[BasePartition]] = None
        self.stability_profile_: Optional[SampleStabilityProfile] = None
        self.consensus_matrix_: Optional[np.ndarray] = None
        self.dissimilarity_matrix_: Optional[np.ndarray] = None
        self.fingerprints_: Optional[List[ClusterFingerprint]] = None
        self.consensus_engine_: Optional[ErrorAwareConsensus] = None

        # Convenience aliases
        self.core_mask_: Optional[np.ndarray] = None
        self.halo_mask_: Optional[np.ndarray] = None
        self.uncertain_mask_: Optional[np.ndarray] = None
        self.stability_scores_: Optional[np.ndarray] = None
        self.fit_time_: float = 0.0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "NoiseAwareConsensusClustering":
        """Fits the NACC framework on input data matrix X."""
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        t0 = time.perf_counter()

        # Step 1: GA/PSO Feature Optimization (PDF Section 5)
        if self.feature_optimizer is not None:
            weights, cand_idx, opt_stats = optimize_feature_weights(
                X,
                n_clusters=self.n_clusters,
                method=self.feature_optimizer,
                max_candidate_features=min(D, 250),
                max_iterations=self.max_optimization_iter,
                random_state=self.random_state,
            )
            self.feature_weights_ = weights
            self.optimization_stats_ = opt_stats
            # Retain high-weight active features
            threshold = np.percentile(weights[cand_idx], 35)
            self.selected_features_ = cand_idx[weights[cand_idx] >= threshold]
            if len(self.selected_features_) < 3:
                self.selected_features_ = cand_idx[:max(3, min(len(cand_idx), 20))]
        else:
            self.feature_weights_ = np.ones(D, dtype=np.float64)
            self.selected_features_ = np.arange(D)

        # Step 2: Noise-Aware Base Clustering (PDF Section 4)
        base_clusterer = NoiseAwareBaseClusterer(
            n_clusters=self.n_clusters,
            n_base_models=self.n_base_models,
            subspace_ratio=self.subspace_ratio,
            random_state=self.random_state,
        )
        partitions = base_clusterer.generate_views(X, feature_weights=self.feature_weights_)
        self.base_partitions_ = partitions

        # Step 3: Sample Stability and Core/Halo Modeling (PDF Section 6)
        partition_labels_list = [p.labels for p in partitions]
        stability_profile = analyze_sample_stability(
            partition_labels_list,
            tau_core=self.tau_core,
            tau_uncertain=self.tau_uncertain,
        )
        self.stability_profile_ = stability_profile
        self.core_mask_ = stability_profile.core_mask
        self.halo_mask_ = stability_profile.halo_mask
        self.uncertain_mask_ = stability_profile.uncertain_mask
        self.stability_scores_ = stability_profile.stability_scores

        # Step 4: Error-Aware Consensus Fusion (PDF Section 7)
        consensus_engine = ErrorAwareConsensus(
            n_clusters=self.n_clusters,
            dissimilarity_alpha=self.dissimilarity_alpha,
            random_state=self.random_state,
        )
        final_labels = consensus_engine.fuse(
            partitions=partitions,
            stability_profile=stability_profile,
            X=X,
            active_features=self.selected_features_,
        )
        self.labels_ = final_labels
        self.consensus_matrix_ = consensus_engine.consensus_matrix_
        self.dissimilarity_matrix_ = consensus_engine.dissimilarity_matrix_
        self.consensus_engine_ = consensus_engine

        # Step 5: Cluster Reliability Profiling (PDF Section 8)
        if self.profiling_repeats > 0:
            profiler = ClusterReliabilityProfiler(
                n_repeats=self.profiling_repeats,
                random_state=self.random_state,
            )
            # Cluster prediction lambda on active subspace
            active_feats = self.selected_features_

            def fast_cluster(X_in: np.ndarray) -> np.ndarray:
                from sklearn.cluster import MiniBatchKMeans
                sub = X_in[:, active_feats] if active_feats.max() < X_in.shape[1] else X_in
                km = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    batch_size=min(128, len(X_in)),
                    n_init=2,
                )
                return km.fit_predict(sub)

            self.fingerprints_ = profiler.profile(X, final_labels, fast_cluster)

        self.fit_time_ = time.perf_counter() - t0
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fits the NACC model and returns the final consensus cluster labels."""
        self.fit(X, y)
        return self.labels_

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """Inductive out-of-sample projection for new unseen query samples."""
        if self.consensus_engine_ is None:
            raise ValueError("Model must be fitted before predict() can be called.")
        return self.consensus_engine_.predict(X_query)

    def get_summary_dict(self) -> Dict[str, Union[float, int, str]]:
        """Returns executive summary metrics of the fitted NACC model."""
        if self.labels_ is None:
            return {}

        core_pct = float(np.mean(self.core_mask_)) * 100.0 if self.core_mask_ is not None else 0.0
        halo_pct = float(np.mean(self.halo_mask_)) * 100.0 if self.halo_mask_ is not None else 0.0
        unc_pct = float(np.mean(self.uncertain_mask_)) * 100.0 if self.uncertain_mask_ is not None else 0.0

        n_active = len(self.selected_features_) if self.selected_features_ is not None else 0
        mean_stab = float(np.mean(self.stability_scores_)) if self.stability_scores_ is not None else 0.0

        avg_fingerprint = (
            float(np.mean([fp.overall_reliability for fp in self.fingerprints_]))
            if self.fingerprints_
            else 0.0
        )

        return {
            "n_clusters": self.n_clusters,
            "n_base_models": self.n_base_models,
            "core_percentage": round(core_pct, 2),
            "halo_percentage": round(halo_pct, 2),
            "uncertain_percentage": round(unc_pct, 2),
            "mean_sample_stability": round(mean_stab, 4),
            "active_features_count": n_active,
            "mean_cluster_reliability": round(avg_fingerprint, 4),
            "fit_time_seconds": round(self.fit_time_, 3),
        }


# Public alias
NACC = NoiseAwareConsensusClustering
