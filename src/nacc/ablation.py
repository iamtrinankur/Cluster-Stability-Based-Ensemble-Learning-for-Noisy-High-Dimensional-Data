"""Formal 6-Stage Ablation Ladder.

Implements Section 10 of the NACC framework:
Evaluates progressive contributions of each architectural component:
1. Conventional clustering (Standard K-Means)
2. Noise-aware base clustering (Diverse views, uniform co-association)
3. Noise-aware clustering + GA/PSO feature optimization
4. + Sample stability (Core/Halo modeling)
5. + Error-aware consensus (H = sum wm (Rm * Am) / sum wm)
6. Full NACC (Dissimilarity regularization + reliability profiler)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

from src.nacc.base_clustering import NoiseAwareBaseClusterer, create_connectivity_matrix
from src.nacc.metrics import evaluate_all
from src.nacc.nacc import NoiseAwareConsensusClustering
from src.nacc.optimizers import optimize_feature_weights
from src.nacc.stability import analyze_sample_stability


def run_ablation_ladder(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    n_clusters: int = 3,
    n_base_models: int = 15,
    random_state: int = 42,
) -> List[Dict[str, Union[str, float, int]]]:
    """Executes the 6-stage ablation ladder and returns comparative metrics for each stage.

    Returns:
        List of dictionaries with stage name, description, ARI, NMI, Silhouette, and Dunn index.
    """
    N, D = X.shape
    results: List[Dict[str, Union[str, float, int]]] = []

    # -------------------------------------------------------------
    # Stage 1: Conventional Clustering (Standard K-Means)
    # -------------------------------------------------------------
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    pred_1 = km.fit_predict(X)
    scores_1 = evaluate_all(y_true, pred_1, X)
    results.append({
        "stage": 1,
        "name": "Conventional Clustering",
        "description": "Standard Euclidean K-Means (Lloyd 1982)",
        **scores_1,
    })

    # -------------------------------------------------------------
    # Stage 2: Noise-Aware Base Clustering (Diverse views, uniform average)
    # -------------------------------------------------------------
    base_gen = NoiseAwareBaseClusterer(
        n_clusters=n_clusters,
        n_base_models=n_base_models,
        random_state=random_state,
    )
    partitions_2 = base_gen.generate_views(X, feature_weights=None)
    # Simple uniform co-association
    coassoc_2 = np.zeros((N, N), dtype=np.float64)
    for p in partitions_2:
        coassoc_2 += p.connectivity
    coassoc_2 /= len(partitions_2)

    dist_2 = 1.0 - coassoc_2
    np.fill_diagonal(dist_2, 0.0)
    pred_2 = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(dist_2)

    scores_2 = evaluate_all(y_true, pred_2, X)
    results.append({
        "stage": 2,
        "name": "Noise-Aware Base Clustering",
        "description": "Diverse views (DIVI gating, random projections, perturbations)",
        **scores_2,
    })

    # -------------------------------------------------------------
    # Stage 3: Noise-Aware Base Clustering + GA/PSO Feature Optimization
    # -------------------------------------------------------------
    weights_3, cand_3, _ = optimize_feature_weights(
        X,
        n_clusters=n_clusters,
        method="ga",
        max_candidate_features=min(D, 120),
        max_iterations=8,
        random_state=random_state,
    )
    partitions_3 = base_gen.generate_views(X, feature_weights=weights_3)
    coassoc_3 = np.zeros((N, N), dtype=np.float64)
    for p in partitions_3:
        coassoc_3 += p.connectivity
    coassoc_3 /= len(partitions_3)

    dist_3 = 1.0 - coassoc_3
    np.fill_diagonal(dist_3, 0.0)
    pred_3 = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(dist_3)

    scores_3 = evaluate_all(y_true, pred_3, X)
    results.append({
        "stage": 3,
        "name": "+ GA/PSO Feature Opt",
        "description": "Multi-view clustering with GA-optimized feature weights F(W)",
        **scores_3,
    })

    # -------------------------------------------------------------
    # Stage 4: + Sample Stability (Core/Halo Modeling)
    # -------------------------------------------------------------
    labels_list_4 = [p.labels for p in partitions_3]
    stab_4 = analyze_sample_stability(labels_list_4, tau_core=0.65, tau_uncertain=0.35)

    # Core-guided clustering: cluster core samples, assign halo/uncertain to nearest
    core_idx_4 = np.where(stab_4.core_mask)[0]
    if len(core_idx_4) < n_clusters * 2:
        core_idx_4 = np.arange(N)

    sub_dist_4 = dist_3[np.ix_(core_idx_4, core_idx_4)]
    core_pred_4 = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(sub_dist_4)

    pred_4 = np.zeros(N, dtype=int)
    pred_4[core_idx_4] = core_pred_4

    non_core_idx_4 = np.where(~stab_4.core_mask)[0]
    for nc in non_core_idx_4:
        aff = [
            np.mean(coassoc_3[nc, core_idx_4[core_pred_4 == k]])
            for k in range(n_clusters)
        ]
        pred_4[nc] = int(np.argmax(aff))

    scores_4 = evaluate_all(y_true, pred_4, X)
    results.append({
        "stage": 4,
        "name": "+ Sample Stability",
        "description": "Sample stability diagnostics and Core/Halo/Uncertain partitioning",
        **scores_4,
    })

    # -------------------------------------------------------------
    # Stage 5: + Error-Aware Consensus
    # -------------------------------------------------------------
    # Weighted consensus H = sum wm (Rm * Am) / sum wm (without dissimilarity penalty)
    weights_5 = np.array([p.reliability_weight for p in partitions_3])
    R_m_5 = stab_4.reliability_matrix
    H_5 = np.zeros((N, N), dtype=np.float64)

    for m, p in enumerate(partitions_3):
        H_5 += weights_5[m] * (R_m_5 * p.connectivity)
    H_5 /= np.sum(weights_5)
    np.fill_diagonal(H_5, 1.0)
    H_5 = 0.5 * (H_5 + H_5.T)

    sc_5 = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", random_state=random_state, assign_labels="kmeans"
    )
    try:
        pred_5 = sc_5.fit_predict(H_5)
    except Exception:
        pred_5 = pred_4

    scores_5 = evaluate_all(y_true, pred_5, X)
    results.append({
        "stage": 5,
        "name": "+ Error-Aware Consensus",
        "description": "Reliability-weighted connectivity fusion H = sum wm(Rm*Am)/sum wm",
        **scores_5,
    })

    # -------------------------------------------------------------
    # Stage 6: Full NACC (Proposed Framework)
    # -------------------------------------------------------------
    nacc_model = NoiseAwareConsensusClustering(
        n_clusters=n_clusters,
        n_base_models=n_base_models,
        feature_optimizer="ga",
        dissimilarity_alpha=0.15,
        max_optimization_iter=8,
        profiling_repeats=3,
        random_state=random_state,
    )
    pred_6 = nacc_model.fit_predict(X)
    scores_6 = evaluate_all(y_true, pred_6, X)
    results.append({
        "stage": 6,
        "name": "Full NACC (Proposed)",
        "description": "Complete framework with dissimilarity regularization & reliability profiler",
        **scores_6,
    })

    return results
