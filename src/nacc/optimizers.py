"""GA and PSO Feature Optimizers with Standardized Bimodality Screening & Anti-Collapse Entropy.

Implements Section 5 of the NACC framework (Refactored for Rigorous High-Dimensional Optimization):
- Replaces raw variance pre-screening with standardized Sarle's Bimodality Coefficient screening.
- Implements cluster-size entropy penalty to prevent trivial single-cluster collapse.
- Optimizes feature weight vector W maximizing:
    F(W) = lambda_1 * Stability + lambda_2 * Separation
         - lambda_3 * Noise_Sensitivity - lambda_4 * Feature_Complexity
         + lambda_5 * Cluster_Entropy_Balance
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph


def compute_feature_bimodality_score(x: np.ndarray) -> float:
    """Computes Sarle's Bimodality Coefficient on a 1D feature array."""
    n = len(x)
    if n < 4:
        return 0.0
    var = np.var(x)
    if var < 1e-9:
        return 0.0

    skew = float(stats.skew(x, bias=False))
    kurt_excess = float(stats.kurtosis(x, fisher=True, bias=False))

    denominator = kurt_excess + (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    if denominator <= 1e-4:
        denominator = 0.1
    bc = (skew ** 2 + 1.0) / denominator
    return float(np.clip(bc, 0.0, 1.0))


def screen_candidate_features(
    X: np.ndarray,
    max_candidates: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardized unsupervised feature screening for high-dimensional regimes (D >> N).

    Combines:
    1. Sarle's Bimodality Coefficient: identifies features whose distributions split into phenotypic states.
    2. Unsupervised Manifold Laplacian Score: identifies features preserving sample neighborhood topology.

    Returns:
        candidate_indices: Sorted array of selected feature column indices.
        candidate_scores: Normalized screening scores for the candidate pool.
    """
    N, D = X.shape
    if D <= max_candidates:
        return np.arange(D), np.ones(D)

    # 1. Vectorized Sarle's Bimodality Coefficient
    skew = stats.skew(X, axis=0, bias=False)
    kurt_excess = stats.kurtosis(X, axis=0, fisher=True, bias=False)
    denom = kurt_excess + (3.0 * (N - 1) ** 2) / ((N - 2) * (N - 3))
    denom = np.maximum(denom, 0.1)
    bc_scores = np.clip((skew ** 2 + 1.0) / denom, 0.0, 1.0)

    # 2. Vectorized Unsupervised Manifold Laplacian Score
    k_neighbors = min(7, max(3, N // 5))
    A = kneighbors_graph(X, n_neighbors=k_neighbors, mode="connectivity", include_self=False).toarray()
    W = np.maximum(A, A.T)
    d = np.sum(W, axis=1)
    d_sum = np.sum(d)
    L = np.diag(d) - W

    col_means = (d @ X) / max(d_sum, 1e-9)
    X_tilde = X - col_means[None, :]
    num = np.sum(X_tilde * (L @ X_tilde), axis=0)
    den = np.sum(d[:, None] * (X_tilde ** 2), axis=0)
    lap_scores = num / np.maximum(den, 1e-9)

    # 3. Composite unsupervised score
    norm_bc = (bc_scores - np.min(bc_scores)) / (np.ptp(bc_scores) + 1e-9)
    norm_lap = 1.0 - (lap_scores - np.min(lap_scores)) / (np.ptp(lap_scores) + 1e-9)
    composite_scores = 0.5 * norm_bc + 0.5 * norm_lap

    # Select top features
    top_indices = np.argsort(composite_scores)[-max_candidates:]
    candidate_indices = np.sort(top_indices)
    candidate_scores = composite_scores[candidate_indices]

    return candidate_indices, candidate_scores


def evaluate_feature_fitness(
    W_active: np.ndarray,
    X_candidate: np.ndarray,
    n_clusters: int = 3,
    l1_stab: float = 1.0,
    l2_sep: float = 0.8,
    l3_sens: float = 0.6,
    l4_comp: float = 0.2,
    l5_entropy: float = 0.5,
    noise_std: float = 0.10,
    random_state: int = 42,
) -> float:
    """Evaluates the robustness objective F(W) with anti-collapse cluster entropy.

    F(W) = l1 * Stability + l2 * Separation - l3 * Noise_Sensitivity - l4 * Complexity + l5 * Entropy_Balance
    """
    N, P = X_candidate.shape
    weight_sum = np.sum(W_active)
    if weight_sum < 1e-4:
        return -999.0

    # Modulate candidate features by continuous weights W
    X_w = X_candidate * W_active[None, :]

    rng = np.random.default_rng(random_state)

    # 1. Cluster entropy balance (prevents trivial single-cluster collapse)
    try:
        km_full = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state, batch_size=min(128, N), n_init=2
        )
        full_pred = km_full.fit_predict(X_w)
        _, counts = np.unique(full_pred, return_counts=True)
        if len(counts) < n_clusters:
            # Missing cluster penalization
            entropy_score = 0.0
        else:
            probs = counts / np.sum(counts)
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(n_clusters)
            entropy_score = float(np.clip(entropy, 0.0, 1.0))
    except Exception:
        return -999.0

    # 2. Stability across repeated 80% subsampling
    sub_size = max(int(0.8 * N), n_clusters * 2)
    idx1 = rng.choice(N, size=sub_size, replace=False)
    idx2 = rng.choice(N, size=sub_size, replace=False)
    overlap = np.intersect1d(idx1, idx2)

    if len(overlap) < n_clusters * 2:
        stability = 0.0
    else:
        seed1 = int(rng.integers(1, 10000))
        seed2 = int(rng.integers(1, 10000))
        km1 = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed1, batch_size=min(128, sub_size), n_init=2)
        km2 = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed2, batch_size=min(128, sub_size), n_init=2)

        pred1 = km1.fit_predict(X_w[idx1])
        pred2 = km2.fit_predict(X_w[idx2])

        m1 = {val: pred1[i] for i, val in enumerate(idx1) if val in overlap}
        m2 = {val: pred2[i] for i, val in enumerate(idx2) if val in overlap}

        labels1 = np.array([m1[o] for o in overlap])
        labels2 = np.array([m2[o] for o in overlap])
        stability = max(0.0, float(adjusted_rand_score(labels1, labels2)))

    # 3. Separation in weighted subspace
    try:
        ch = calinski_harabasz_score(X_w, full_pred)
        separation = float(np.tanh(ch / 250.0))
    except Exception:
        separation = 0.0

    # 4. Noise sensitivity
    try:
        noise = rng.normal(0.0, noise_std, size=X_w.shape)
        km_noisy = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=min(128, N), n_init=2)
        noisy_pred = km_noisy.fit_predict(X_w + noise)
        sens_ari = float(adjusted_rand_score(full_pred, noisy_pred))
        noise_sensitivity = max(0.0, 1.0 - sens_ari)
    except Exception:
        noise_sensitivity = 1.0

    # 5. Feature complexity (L1 sparsity)
    feature_complexity = float(np.mean(W_active))

    # Unified objective
    fitness = (
        l1_stab * stability
        + l2_sep * separation
        - l3_sens * noise_sensitivity
        - l4_comp * feature_complexity
        + l5_entropy * entropy_score
    )

    return float(fitness)


class GeneticAlgorithmFeatureOptimizer:
    """Genetic Algorithm (GA) optimizer searching for robust feature weights."""

    def __init__(
        self,
        population_size: int = 20,
        max_generations: int = 15,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15,
        mutation_scale: float = 0.1,
        elitism_count: int = 2,
        lambda_stability: float = 1.0,
        lambda_separation: float = 0.8,
        lambda_noise_sensitivity: float = 0.6,
        lambda_complexity: float = 0.2,
        lambda_entropy: float = 0.5,
        random_state: int = 42,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elitism_count = elitism_count
        self.lambda_stability = lambda_stability
        self.lambda_separation = lambda_separation
        self.lambda_noise_sensitivity = lambda_noise_sensitivity
        self.lambda_complexity = lambda_complexity
        self.lambda_entropy = lambda_entropy
        self.random_state = random_state

        self.best_weights_: Optional[np.ndarray] = None
        self.best_fitness_: float = -np.inf
        self.history_: List[float] = []

    def optimize(
        self,
        X_candidate: np.ndarray,
        n_clusters: int = 3,
        initial_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Runs Genetic Algorithm to find optimal feature weights W in [0, 1]^P."""
        N, P = X_candidate.shape
        rng = np.random.default_rng(self.random_state)

        # Initialize population
        population = rng.uniform(0.1, 0.9, size=(self.population_size, P))
        if initial_weights is not None:
            population[0] = np.clip(initial_weights, 0.0, 1.0)
            population[1] = np.where(initial_weights > np.median(initial_weights), 1.0, 0.1)

        best_weights = population[0].copy()
        best_fitness = -np.inf

        for gen in range(self.max_generations):
            fitness_scores = np.zeros(self.population_size)
            for i in range(self.population_size):
                fitness_scores[i] = evaluate_feature_fitness(
                    population[i],
                    X_candidate,
                    n_clusters=n_clusters,
                    l1_stab=self.lambda_stability,
                    l2_sep=self.lambda_separation,
                    l3_sens=self.lambda_noise_sensitivity,
                    l4_comp=self.lambda_complexity,
                    l5_entropy=self.lambda_entropy,
                    random_state=self.random_state + gen * 100 + i,
                )

            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_weights = population[gen_best_idx].copy()

            self.history_.append(float(best_fitness))

            # Elitism
            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_population = [population[idx].copy() for idx in sorted_indices[: self.elitism_count]]

            # Reproduction
            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, fitness_scores, rng)
                p2 = self._tournament_select(population, fitness_scores, rng)

                if rng.random() < self.crossover_rate:
                    alpha = rng.uniform(0.2, 0.8)
                    child = alpha * p1 + (1.0 - alpha) * p2
                else:
                    child = p1.copy()

                mutation_mask = rng.random(P) < self.mutation_rate
                if np.any(mutation_mask):
                    noise = rng.normal(0.0, self.mutation_scale, size=P)
                    child[mutation_mask] += noise[mutation_mask]

                child = np.clip(child, 0.0, 1.0)
                new_population.append(child)

            population = np.array(new_population)

        self.best_weights_ = best_weights
        self.best_fitness_ = best_fitness
        return best_weights

    def _tournament_select(
        self, population: np.ndarray, fitness_scores: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        contestants = rng.choice(len(population), size=self.tournament_size, replace=False)
        winner_idx = contestants[np.argmax(fitness_scores[contestants])]
        return population[winner_idx]


class PSOFeatureOptimizer:
    """Particle Swarm Optimization (PSO) for feature weight optimization."""

    def __init__(
        self,
        n_particles: int = 20,
        max_iterations: int = 15,
        w: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4,
        lambda_stability: float = 1.0,
        lambda_separation: float = 0.8,
        lambda_noise_sensitivity: float = 0.6,
        lambda_complexity: float = 0.2,
        lambda_entropy: float = 0.5,
        random_state: int = 42,
    ):
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lambda_stability = lambda_stability
        self.lambda_separation = lambda_separation
        self.lambda_noise_sensitivity = lambda_noise_sensitivity
        self.lambda_complexity = lambda_complexity
        self.lambda_entropy = lambda_entropy
        self.random_state = random_state

        self.best_weights_: Optional[np.ndarray] = None
        self.best_fitness_: float = -np.inf
        self.history_: List[float] = []

    def optimize(
        self,
        X_candidate: np.ndarray,
        n_clusters: int = 3,
        initial_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        N, P = X_candidate.shape
        rng = np.random.default_rng(self.random_state)

        positions = rng.uniform(0.1, 0.9, size=(self.n_particles, P))
        velocities = rng.uniform(-0.1, 0.1, size=(self.n_particles, P))

        if initial_weights is not None:
            positions[0] = np.clip(initial_weights, 0.0, 1.0)

        personal_best_pos = positions.copy()
        personal_best_fit = np.full(self.n_particles, -np.inf)

        global_best_pos = positions[0].copy()
        global_best_fit = -np.inf

        for it in range(self.max_iterations):
            for i in range(self.n_particles):
                fit = evaluate_feature_fitness(
                    positions[i],
                    X_candidate,
                    n_clusters=n_clusters,
                    l1_stab=self.lambda_stability,
                    l2_sep=self.lambda_separation,
                    l3_sens=self.lambda_noise_sensitivity,
                    l4_comp=self.lambda_complexity,
                    l5_entropy=self.lambda_entropy,
                    random_state=self.random_state + it * 100 + i,
                )

                if fit > personal_best_fit[i]:
                    personal_best_fit[i] = fit
                    personal_best_pos[i] = positions[i].copy()

                if fit > global_best_fit:
                    global_best_fit = fit
                    global_best_pos = positions[i].copy()

            self.history_.append(float(global_best_fit))

            r1 = rng.uniform(0.0, 1.0, size=(self.n_particles, P))
            r2 = rng.uniform(0.0, 1.0, size=(self.n_particles, P))

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (personal_best_pos - positions)
                + self.c2 * r2 * (global_best_pos[None, :] - positions)
            )
            velocities = np.clip(velocities, -0.2, 0.2)
            positions = np.clip(positions + velocities, 0.0, 1.0)

        self.best_weights_ = global_best_pos
        self.best_fitness_ = global_best_fit
        return global_best_pos


def optimize_feature_weights(
    X: np.ndarray,
    n_clusters: int = 3,
    method: str = "ga",
    max_candidate_features: int = 150,
    max_iterations: int = 12,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """High-level wrapper that performs standardized bimodality screening and GA/PSO optimization."""
    N, D = X.shape

    # 1. Standardized unsupervised bimodality screening (Action A4)
    candidate_idx, cand_scores = screen_candidate_features(
        X, max_candidates=min(D, max_candidate_features)
    )

    X_cand = X[:, candidate_idx]
    init_weights = 0.3 + 0.6 * cand_scores

    if method.lower() == "pso":
        optimizer = PSOFeatureOptimizer(
            max_iterations=max_iterations,
            random_state=random_state,
        )
    else:
        optimizer = GeneticAlgorithmFeatureOptimizer(
            max_generations=max_iterations,
            random_state=random_state,
        )

    best_cand_weights = optimizer.optimize(
        X_cand, n_clusters=n_clusters, initial_weights=init_weights
    )

    full_weights = np.full(D, 0.05, dtype=np.float64)
    full_weights[candidate_idx] = best_cand_weights

    stats = {
        "best_fitness": round(optimizer.best_fitness_, 4),
        "mean_active_weight": round(float(np.mean(best_cand_weights)), 4),
        "n_candidates": len(candidate_idx),
        "optimizer": method,
    }

    return full_weights, candidate_idx, stats
