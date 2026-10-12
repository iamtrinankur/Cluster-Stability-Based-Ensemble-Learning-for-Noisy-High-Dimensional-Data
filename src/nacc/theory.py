"""Theoretical Spectral Analysis and Eigengap Verification.

Implements Plan B: Formal Theoretical Foundations of NACC
Provides mathematical formulations, theorem statements, and empirical simulation
demonstrating the Spectral Eigengap Expansion Theorem.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import linalg


def compute_spectral_gap(affinity_matrix: np.ndarray, k_clusters: int) -> Tuple[float, np.ndarray]:
    """Computes the normalized Laplacian spectral eigengap Delta = lambda_{k+1} - lambda_k.

    For normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}:
    Eigenvalues are in [0, 2].
    A large gap Delta between lambda_{k+1} and lambda_k indicates strong cluster separation
    according to Davis-Kahan perturbation theory.
    """
    N = affinity_matrix.shape[0]
    degrees = np.sum(affinity_matrix, axis=1)
    degrees = np.where(degrees < 1e-9, 1e-9, degrees)
    inv_sqrt_d = 1.0 / np.sqrt(degrees)

    # L_sym = I - D^{-1/2} A D^{-1/2}
    L_sym = np.eye(N) - (inv_sqrt_d[:, None] * affinity_matrix * inv_sqrt_d[None, :])
    L_sym = 0.5 * (L_sym + L_sym.T)

    eigenvalues = linalg.eigvalsh(L_sym)
    eigenvalues = np.sort(np.maximum(0.0, eigenvalues))

    if k_clusters < len(eigenvalues):
        # Davis-Kahan eigengap
        gap = float(eigenvalues[k_clusters] - eigenvalues[k_clusters - 1])
    else:
        gap = 0.0

    return gap, eigenvalues


def verify_spectral_eigengap_expansion(
    n_samples: int = 60,
    k_clusters: int = 3,
    noise_ratio: float = 0.35,
    n_trials: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Empirically verifies Theorem 1 (Spectral Eigengap Expansion).

    Theorem Statement:
    Let A_bar be an unweighted co-association matrix subject to inter-cluster noise perturbation E:
        A_bar = A_ideal + E_noise
    Under the error-aware consensus operator:
        H_NACC = H_rel * (1 - alpha * D_dissim^2)
    where D_dissim = 1 - A_bar and H_rel = (1 - gamma)*A_bar + gamma*(R * A_bar),
    inter-cluster noise edges are quadratically suppressed by (1 - alpha * D^2) while
    intra-cluster core connections are preserved, strictly expanding the spectral gap
    Delta_NACC > Delta_std with high probability.
    """
    rng = np.random.default_rng(random_state)
    block_size = n_samples // k_clusters

    standard_gaps = []
    nacc_gaps = []

    for trial in range(n_trials):
        seed = random_state + trial * 13
        t_rng = np.random.default_rng(seed)

        # 1. Ideal block diagonal affinity
        A_ideal = np.zeros((n_samples, n_samples))
        for k in range(k_clusters):
            idx = slice(k * block_size, (k + 1) * block_size)
            A_ideal[idx, idx] = 1.0

        # 2. Add inter-cluster noise perturbation edges
        noise_mask = t_rng.random((n_samples, n_samples)) < noise_ratio
        noise_mat = 0.5 * (noise_mask + noise_mask.T) * t_rng.uniform(0.1, 0.5, size=(n_samples, n_samples))
        np.fill_diagonal(noise_mat, 0.0)

        A_noisy = np.clip(A_ideal + noise_mat, 0.0, 1.0)
        np.fill_diagonal(A_noisy, 1.0)

        # 3. Standard Consensus matrix
        gap_std, _ = compute_spectral_gap(A_noisy, k_clusters)

        # 4. NACC Reliability & Dissimilarity Regularized Matrix
        stability = np.ones(n_samples)
        for k in range(k_clusters):
            b_start = k * block_size + int(0.75 * block_size)
            b_end = (k + 1) * block_size
            stability[b_start:b_end] = 0.40  # Boundary samples

        R = np.sqrt(stability[:, None] * stability[None, :])
        gamma = 0.40
        alpha = 0.35

        H_rel = (1.0 - gamma) * A_noisy + gamma * (R * A_noisy)
        D_dissim = 1.0 - A_noisy
        H_nacc = H_rel * (1.0 - alpha * (D_dissim ** 2))
        np.fill_diagonal(H_nacc, 1.0)
        H_nacc = 0.5 * (H_nacc + H_nacc.T)

        gap_nacc, _ = compute_spectral_gap(H_nacc, k_clusters)

        standard_gaps.append(gap_std)
        nacc_gaps.append(gap_nacc)

    mean_std = float(np.mean(standard_gaps))
    mean_nacc = float(np.mean(nacc_gaps))
    expansion_ratio = float(mean_nacc / (mean_std + 1e-12))

    return {
        "mean_gap_standard": round(mean_std, 4),
        "mean_gap_nacc": round(mean_nacc, 4),
        "eigengap_expansion_ratio": round(expansion_ratio, 3),
        "expansion_confirmed": bool(mean_nacc > mean_std),
    }
