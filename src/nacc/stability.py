"""Sample Stability and Core/Halo/Uncertain Modeling.

Implements Section 6 of the NACC framework (Refactored for Alignment-Free Permutation Invariance):
- Measures sample-level stability strictly from co-association determinacy and contrast.
- Eliminates reference-partition label matching bias.
- Categorizes instances into:
  * Core: consistently assigned and strongly supported.
  * Halo: close to cluster boundaries or moderately stable.
  * Uncertain: frequently fluctuating or weakly supported.
- Generates degree-preserving sample reliability matrices R_m.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class SampleStabilityProfile:
    """Encapsulates sample stability diagnostics and core/halo masks."""
    stability_scores: np.ndarray  # (N,) continuous score in [0, 1]
    core_mask: np.ndarray  # (N,) boolean
    halo_mask: np.ndarray  # (N,) boolean
    uncertain_mask: np.ndarray  # (N,) boolean
    sample_categories: np.ndarray  # (N,) string array: 'Core', 'Halo', 'Uncertain'
    reliability_matrix: np.ndarray  # (N, N) pairwise reliability weights
    core_ratio: float
    halo_ratio: float
    uncertain_ratio: float


def align_partition_labels(y_ref: np.ndarray, y_target: np.ndarray) -> np.ndarray:
    """Aligns target cluster labels to reference labels using the Hungarian algorithm.
    Retained for diagnostic reporting and partition correspondence.
    """
    y_ref = np.asarray(y_ref)
    y_target = np.asarray(y_target)

    ref_clusters = np.unique(y_ref)
    target_clusters = np.unique(y_target)

    cost_matrix = np.zeros((len(ref_clusters), len(target_clusters)))
    for i, r in enumerate(ref_clusters):
        for j, t in enumerate(target_clusters):
            cost_matrix[i, j] = -np.sum((y_ref == r) & (y_target == t))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {target_clusters[c]: ref_clusters[r] for r, c in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(lbl, lbl) for lbl in y_target])
    return aligned


def analyze_sample_stability(
    base_partitions: List[np.ndarray],
    tentative_labels: Optional[np.ndarray] = None,
    tau_core: float = 0.65,
    tau_uncertain: float = 0.35,
    k_clusters: Optional[int] = None,
) -> SampleStabilityProfile:
    """Computes alignment-free sample stability scores and Core/Halo/Uncertain stratification.

    Mathematical Formulation:
    Stability is evaluated directly on the ensemble co-association operator A_bar:
        A_bar(i, j) = (1 / M) * sum_{m=1}^M 1{ y_m(i) == y_m(j) }
    For each instance i, determinacy is computed via soft co-association contrast
    and entropy across latent cluster components, guaranteeing 100% permutation invariance.

    Args:
        base_partitions: List of label arrays from base clusterings [labels_1, ..., labels_M]
        tentative_labels: Optional reference labels (if None, extracted via normalized cut)
        tau_core: Threshold for Core classification
        tau_uncertain: Threshold below which samples are categorized as Uncertain
        k_clusters: Number of clusters (if None, inferred from partitions)

    Returns:
        SampleStabilityProfile dataclass.
    """
    M = len(base_partitions)
    N = len(base_partitions[0])
    if k_clusters is None:
        k_clusters = len(np.unique(base_partitions[0]))

    # 1. Compute ensemble co-association matrix (Strictly permutation-invariant)
    coassoc = np.zeros((N, N), dtype=np.float64)
    for labels in base_partitions:
        coassoc += (labels[:, None] == labels[None, :]).astype(np.float64)
    coassoc /= M
    np.fill_diagonal(coassoc, 1.0)

    # 2. Extract cluster membership components from coassoc
    if tentative_labels is None:
        from sklearn.cluster import SpectralClustering
        try:
            sc = SpectralClustering(
                n_clusters=k_clusters,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
                n_init=5,
            )
            tentative_labels = sc.fit_predict(coassoc)
        except Exception:
            tentative_labels = base_partitions[0]

    unique_clusters = np.unique(tentative_labels)
    K = len(unique_clusters)

    # 3. Alignment-Free Co-Association Margin & Soft Entropy
    cluster_masks = [tentative_labels == c for c in unique_clusters]

    stability_scores = np.zeros(N, dtype=np.float64)

    for i in range(N):
        # Mean co-association of sample i with each cluster c
        mu_ic = np.zeros(K, dtype=np.float64)
        for c_idx, mask in enumerate(cluster_masks):
            # Exclude self to avoid trivial self-correlation
            sub_mask = mask.copy()
            sub_mask[i] = False
            if np.sum(sub_mask) > 0:
                mu_ic[c_idx] = np.mean(coassoc[i, sub_mask])
            else:
                mu_ic[c_idx] = coassoc[i, i] if mask[i] else 0.0

        # Margin: difference between best cluster affinity and second best
        sorted_aff = np.sort(mu_ic)
        if K >= 2:
            margin = sorted_aff[-1] - sorted_aff[-2]
        else:
            margin = sorted_aff[-1]

        # Normalized Entropy Contrast:
        # Softmax over affinities
        temperature = 0.2
        exp_aff = np.exp((mu_ic - np.max(mu_ic)) / temperature)
        probs = exp_aff / np.sum(exp_aff)
        entropy = -np.sum(probs * np.log(probs + 1e-12)) / (np.log(K) if K > 1 else 1.0)
        contrast = max(0.0, 1.0 - entropy)

        # Composite stability score in [0, 1]
        score = 0.5 * margin + 0.5 * contrast
        stability_scores[i] = float(np.clip(score, 0.0, 1.0))

    # 4. Stratify into Core, Halo, and Uncertain
    core_mask = stability_scores >= tau_core
    uncertain_mask = stability_scores < tau_uncertain
    halo_mask = (~core_mask) & (~uncertain_mask)

    # Safe fallback if thresholds are too strict
    if not np.any(core_mask):
        median_val = np.median(stability_scores)
        core_mask = stability_scores >= median_val
        halo_mask = ~core_mask
        uncertain_mask = np.zeros(N, dtype=bool)

    categories = np.empty(N, dtype=object)
    categories[core_mask] = "Core"
    categories[halo_mask] = "Halo"
    categories[uncertain_mask] = "Uncertain"

    # 5. Degree-Preserving Pairwise Reliability Tensor:
    # R_m(i, j) = sqrt(s_i * s_j) with degree floor epsilon to prevent graph disconnection
    eps_floor = 0.05
    s_floored = eps_floor + (1.0 - eps_floor) * stability_scores
    reliability_matrix = np.sqrt(s_floored[:, None] * s_floored[None, :])

    return SampleStabilityProfile(
        stability_scores=stability_scores,
        core_mask=core_mask,
        halo_mask=halo_mask,
        uncertain_mask=uncertain_mask,
        sample_categories=categories,
        reliability_matrix=reliability_matrix,
        core_ratio=float(np.mean(core_mask)),
        halo_ratio=float(np.mean(halo_mask)),
        uncertain_ratio=float(np.mean(uncertain_mask)),
    )
