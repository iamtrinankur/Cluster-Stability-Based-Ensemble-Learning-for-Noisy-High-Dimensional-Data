"""Noise-Aware Consensus Clustering (NACC) Python Package.

Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data.
"""

from src.nacc.ablation import run_ablation_ladder
from src.nacc.base_clustering import BasePartition, NoiseAwareBaseClusterer, divi_feature_gating
from src.nacc.biology import evaluate_biomarker_enrichment
from src.nacc.consensus import ErrorAwareConsensus
from src.nacc.datasets import generate_synthetic_noisy_data, list_available_datasets, load_dataset
from src.nacc.downstream import evaluate_downstream_utility
from src.nacc.metrics import (
    clustering_purity,
    dunn_index,
    evaluate_all,
    evaluate_external,
    evaluate_internal,
)
from src.nacc.nacc import NACC, NoiseAwareConsensusClustering
from src.nacc.optimizers import (
    GeneticAlgorithmFeatureOptimizer,
    PSOFeatureOptimizer,
    optimize_feature_weights,
    screen_candidate_features,
)
from src.nacc.reliability import ClusterFingerprint, ClusterReliabilityProfiler
from src.nacc.stability import SampleStabilityProfile, align_partition_labels, analyze_sample_stability
from src.nacc.theory import compute_spectral_gap, verify_spectral_eigengap_expansion
from src.nacc.viz import (
    plot_consensus_heatmap,
    plot_noise_response_curves,
    plot_reliability_radar,
    plot_sample_stability_projection,
)

__version__ = "1.0.0"

__all__ = [
    "NACC",
    "NoiseAwareConsensusClustering",
    "BasePartition",
    "NoiseAwareBaseClusterer",
    "divi_feature_gating",
    "GeneticAlgorithmFeatureOptimizer",
    "PSOFeatureOptimizer",
    "optimize_feature_weights",
    "screen_candidate_features",
    "SampleStabilityProfile",
    "analyze_sample_stability",
    "align_partition_labels",
    "ErrorAwareConsensus",
    "ClusterFingerprint",
    "ClusterReliabilityProfiler",
    "evaluate_all",
    "evaluate_internal",
    "evaluate_external",
    "dunn_index",
    "clustering_purity",
    "evaluate_downstream_utility",
    "run_ablation_ladder",
    "load_dataset",
    "list_available_datasets",
    "generate_synthetic_noisy_data",
    "compute_spectral_gap",
    "verify_spectral_eigengap_expansion",
    "evaluate_biomarker_enrichment",
    "plot_consensus_heatmap",
    "plot_reliability_radar",
    "plot_sample_stability_projection",
    "plot_noise_response_curves",
]
