"""Tests for GA and PSO feature optimizers."""

import numpy as np
import pytest
from src.nacc.optimizers import (
    GeneticAlgorithmFeatureOptimizer,
    PSOFeatureOptimizer,
    evaluate_feature_fitness,
    optimize_feature_weights,
)


def test_evaluate_feature_fitness():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(-3, 0.5, size=(30, 6)),
        rng.normal(3, 0.5, size=(30, 6)),
    ])
    W = np.ones(6)
    score = evaluate_feature_fitness(W, X, n_clusters=2)
    assert isinstance(score, float)
    assert not np.isnan(score)


def test_genetic_algorithm_optimizer():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(-3, 0.5, size=(30, 10)),
        rng.normal(3, 0.5, size=(30, 10)),
    ])
    ga = GeneticAlgorithmFeatureOptimizer(
        population_size=10,
        max_generations=4,
        random_state=42,
    )
    best_w = ga.optimize(X, n_clusters=2)
    assert len(best_w) == 10
    assert np.all(best_w >= 0.0) and np.all(best_w <= 1.0)
    assert ga.best_fitness_ > -900.0


def test_pso_optimizer():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(-3, 0.5, size=(30, 10)),
        rng.normal(3, 0.5, size=(30, 10)),
    ])
    pso = PSOFeatureOptimizer(
        n_particles=10,
        max_iterations=4,
        random_state=42,
    )
    best_w = pso.optimize(X, n_clusters=2)
    assert len(best_w) == 10
    assert np.all(best_w >= 0.0) and np.all(best_w <= 1.0)
    assert pso.best_fitness_ > -900.0


def test_optimize_feature_weights_wrapper():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(40, 25))
    weights, cand, stats = optimize_feature_weights(
        X, n_clusters=2, method="ga", max_candidate_features=15, max_iterations=3
    )
    assert len(weights) == 25
    assert len(cand) == 15
    assert "best_fitness" in stats
