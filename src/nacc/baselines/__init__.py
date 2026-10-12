"""Baseline clustering algorithms factory and unified interface."""

from typing import Any, Callable, Dict
import numpy as np

from src.nacc.baselines.classical import run_fuzzy_cmeans, run_kmeans, run_spectral, run_ward
from src.nacc.baselines.consensus import CoreHaloConsensusClustering, StandardConsensusClustering


def get_baseline_runner(name: str) -> Callable[[np.ndarray, int, int], np.ndarray]:
    """Returns a callable f(X, n_clusters, random_state) -> labels for the requested baseline."""
    name_clean = name.lower().replace("-", "_").replace(" ", "_")

    if name_clean in ["kmeans", "k_means"]:
        return lambda X, k, seed: run_kmeans(X, k, random_state=seed)
    elif name_clean in ["ward", "hierarchical"]:
        return lambda X, k, seed: run_ward(X, k)
    elif name_clean in ["spectral", "spectral_clustering"]:
        return lambda X, k, seed: run_spectral(X, k, random_state=seed)
    elif name_clean in ["fuzzy_cmeans", "fcm", "fuzzy"]:
        return lambda X, k, seed: run_fuzzy_cmeans(X, k, random_state=seed)
    elif name_clean in ["standard_consensus", "sc"]:
        return lambda X, k, seed: StandardConsensusClustering(n_clusters=k, random_state=seed).fit_predict(X)
    elif name_clean in ["core_halo", "core_halo_consensus"]:
        return lambda X, k, seed: CoreHaloConsensusClustering(n_clusters=k, random_state=seed).fit_predict(X)
    else:
        raise ValueError(f"Unknown baseline: '{name}'")
