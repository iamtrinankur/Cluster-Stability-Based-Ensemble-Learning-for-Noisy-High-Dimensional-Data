"""Classical Single-Model Clustering Baselines.

Implements standard benchmark algorithms:
- K-Means (Lloyd 1982)
- Ward Hierarchical Agglomerative Linkage (Ward 1963)
- Spectral Clustering (von Luxburg 2007)
- Fuzzy C-Means (Bezdek 1981)
"""

from typing import Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering


def run_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Executes standard K-Means clustering."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(X)


def run_ward(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Executes Ward hierarchical agglomerative clustering."""
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return ward.fit_predict(X)


def run_spectral(X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Executes Normalized Spectral Clustering with RBF graph affinity."""
    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(15, max(3, len(X) - 1)),
            random_state=random_state,
            assign_labels="kmeans",
            n_init=5,
        )
        return sc.fit_predict(X)
    except Exception:
        # Fallback to K-Means if graph fails to construct
        return run_kmeans(X, n_clusters, random_state)


def run_fuzzy_cmeans(
    X: np.ndarray,
    n_clusters: int,
    m: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
) -> np.ndarray:
    """Executes Fuzzy C-Means (Bezdek 1981) returning hard defuzzified cluster assignments."""
    rng = np.random.default_rng(random_state)
    N, D = X.shape

    # Initialize random soft membership matrix U (N, K)
    U = rng.uniform(0.01, 1.0, size=(N, n_clusters))
    U = U / np.sum(U, axis=1, keepdims=True)

    for _ in range(max_iter):
        U_prev = U.copy()
        # Compute cluster centers
        um = U ** m  # (N, K)
        centers = (um.T @ X) / (np.sum(um, axis=0, keepdims=True).T + 1e-12)  # (K, D)

        # Compute Euclidean distance from each sample to centers
        dists = np.zeros((N, n_clusters))
        for k in range(n_clusters):
            dists[:, k] = np.linalg.norm(X - centers[k], axis=1) + 1e-9

        # Update memberships U
        power = 2.0 / (m - 1.0)
        inv_dists = (1.0 / dists) ** power
        U = inv_dists / (np.sum(inv_dists, axis=1, keepdims=True) + 1e-12)

        if np.max(np.abs(U - U_prev)) < tol:
            break

    # Hard assignment by maximum membership
    return np.argmax(U, axis=1)
