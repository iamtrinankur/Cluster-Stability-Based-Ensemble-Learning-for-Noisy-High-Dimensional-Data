#!/usr/bin/env python3
"""Noise-Aware Consensus Clustering (NACC) — Interactive Walkthrough Demo.

Project: Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data
Demonstrates:
1. High-dimensional noisy data synthesis (D = 2,000, N = 240, K = 3, 99.8% noise features).
2. GA-based multi-objective feature optimization F(W).
3. Noise-aware base clustering across diverse views (DIVI gating, random projections, perturbations).
4. Sample stability analysis and Core / Halo / Uncertain categorization.
5. Error-aware consensus fusion and dissimilarity regularization.
6. Cluster Reliability Profiling (Noise, Dropout, Subsampling, Distribution Shift, Batch Effects).
7. Inductive out-of-sample projection on unseen test queries.
8. Comparative evaluation against K-Means, Spectral Clustering, Standard Consensus, and Core-Halo.

Run:
    python demo.py
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
import time
from pathlib import Path
import numpy as np
from tabulate import tabulate

from src.nacc.baselines.classical import run_kmeans, run_spectral
from src.nacc.baselines.consensus import CoreHaloConsensusClustering, StandardConsensusClustering
from src.nacc.datasets import generate_synthetic_noisy_data
from src.nacc.metrics import evaluate_all
from src.nacc.nacc import NoiseAwareConsensusClustering
from src.nacc.viz import (
    plot_consensus_heatmap,
    plot_reliability_radar,
    plot_sample_stability_projection,
)


def main():
    print("=" * 80)
    print("      NOISE-AWARE CONSENSUS CLUSTERING (NACC) -- INTERACTIVE DEMO")
    print("  Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data")
    print("=" * 80)

    # 1. Synthesize High-Dimensional Cohort
    N_train = 240
    N_test = 60
    D_signal = 4
    D_noise = 1996
    D_total = D_signal + D_noise
    K = 3

    print(f"\n[*] 1. Synthesizing High-Dimensional Noisy Cohort:")
    print(f"    - Training Samples (N_train): {N_train}")
    print(f"    - Query Samples (N_test):     {N_test}")
    print(f"    - True Signal Dimensions:    {D_signal} (Features [0..{D_signal-1}])")
    print(f"    - Distractor Noise Dims:     {D_noise} (Nuisance Features [{D_signal}..{D_total-1}])")
    print(f"    - Noise Ratio:               {D_noise / D_total * 100:.2f}%")
    print(f"    - True Clusters (K):         {K}")

    X_train, y_train, _ = generate_synthetic_noisy_data(
        n_samples=N_train,
        d_signal=D_signal,
        d_noise=D_noise,
        n_clusters=K,
        cluster_std=0.85,
        random_state=42,
    )

    X_test, y_test, _ = generate_synthetic_noisy_data(
        n_samples=N_test,
        d_signal=D_signal,
        d_noise=D_noise,
        n_clusters=K,
        cluster_std=0.85,
        random_state=142,
    )

    # 2. Fit NACC Framework
    print(f"\n[*] 2. Fitting NACC Framework (GA Optimization + Diverse Views + Core/Halo Modeling)...")
    t0 = time.perf_counter()
    nacc = NoiseAwareConsensusClustering(
        n_clusters=K,
        n_base_models=15,
        feature_optimizer="ga",
        subspace_ratio=0.25,
        tau_core=0.65,
        tau_uncertain=0.35,
        dissimilarity_alpha=0.15,
        max_optimization_iter=8,
        profiling_repeats=3,
        random_state=42,
    )
    y_pred_nacc = nacc.fit_predict(X_train)
    fit_time = time.perf_counter() - t0

    # Summary diagnostics
    diag = nacc.get_summary_dict()
    print(f"    -> Optimization & Consensus Time: {fit_time:.2f} s")
    print(f"    -> Certified Core Samples:        {diag['core_percentage']}%")
    print(f"    -> Boundary Halo Samples:         {diag['halo_percentage']}%")
    print(f"    -> Uncertain / Fragile Samples:   {diag['uncertain_percentage']}%")
    print(f"    -> Active Features Retained:      {diag['active_features_count']} of {D_total}")

    # Check signal recovery
    recovered_signals = [i for i in range(D_signal) if i in nacc.selected_features_]
    print(f"    -> True Signal Features Recovered: {recovered_signals} ({len(recovered_signals)}/{D_signal})")

    # 3. Inductive Out-of-Sample Query Prediction
    print(f"\n[*] 3. Inductive Out-of-Sample Projection on Unseen Test Batch...")
    t0_pred = time.perf_counter()
    test_pred_nacc = nacc.predict(X_test)
    pred_time = time.perf_counter() - t0_pred
    print(f"    -> Projection Latency: {pred_time * 1000:.2f} ms for {N_test} samples ({pred_time / N_test * 1e6:.1f} us/sample)")

    # 4. Comparative Evaluation Against Baselines
    print(f"\n[*] 4. Executing Benchmark Baselines (K-Means, Spectral, Standard Consensus, Core-Halo)...")
    models = {
        "K-Means (Lloyd 1982)": run_kmeans(X_train, K, random_state=42),
        "Spectral (von Luxburg)": run_spectral(X_train, K, random_state=42),
        "Standard Consensus": StandardConsensusClustering(n_clusters=K, random_state=42).fit_predict(X_train),
        "Core-Halo (Li et al.)": CoreHaloConsensusClustering(n_clusters=K, random_state=42).fit_predict(X_train),
        "NACC (Proposed Framework)": y_pred_nacc,
    }

    table_data = []
    for model_name, preds in models.items():
        evals = evaluate_all(y_train, preds, X_train)
        table_data.append([
            model_name,
            f"{evals['ari']:.4f}",
            f"{evals['nmi']:.4f}",
            f"{evals['purity']:.4f}",
            f"{evals['silhouette']:.4f}",
            f"{evals['dunn_index']:.4f}",
        ])

    print("\n" + tabulate(
        table_data,
        headers=["Clustering Model", "ARI", "NMI", "Purity", "Silhouette", "Dunn Index"],
        tablefmt="github",
    ))

    # Test set inductive score
    test_evals = evaluate_all(y_test, test_pred_nacc, X_test)
    print(f"\n[+] Out-of-Sample Inductive Generalization (NACC): ARI = {test_evals['ari']:.4f}, NMI = {test_evals['nmi']:.4f}, Purity = {test_evals['purity']:.4f}")

    # 5. Cluster Reliability Fingerprints
    print(f"\n[*] 5. Cluster Reliability Fingerprints (5 Perturbation Regimes):")
    fp_table = []
    for fp in nacc.fingerprints_:
        fp_table.append([
            f"Cluster {fp.cluster_id}",
            fp.size,
            f"{fp.noise_stability:.3f}",
            f"{fp.dropout_stability:.3f}",
            f"{fp.sample_stability:.3f}",
            f"{fp.shift_stability:.3f}",
            f"{fp.batch_stability:.3f}",
            f"{fp.overall_reliability:.3f}",
            fp.stability_grade,
        ])

    print(tabulate(
        fp_table,
        headers=["Cluster", "Size", "Noise", "Dropout", "Subsample", "Shift", "Batch", "Mean Rel", "Grade"],
        tablefmt="github",
    ))

    # 6. Save Visualization Artifacts
    fig_dir = Path("experiments") / "demo_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_consensus_heatmap(
        nacc.consensus_matrix_,
        y_pred_nacc,
        save_path=fig_dir / "demo_consensus_matrix.png",
    )
    plot_reliability_radar(
        nacc.fingerprints_,
        save_path=fig_dir / "demo_reliability_radar.png",
    )
    plot_sample_stability_projection(
        X_train,
        nacc.stability_profile_.sample_categories,
        nacc.stability_scores_,
        save_path=fig_dir / "demo_sample_stability_pca.png",
    )

    print(f"\n[+] Visualizations successfully generated in: {fig_dir.resolve()}")
    print("    - demo_consensus_matrix.png: Block-diagonal error-aware affinity")
    print("    - demo_reliability_radar.png: Multi-regime cluster fingerprints")
    print("    - demo_sample_stability_pca.png: Core vs Halo vs Uncertain projection")
    print("\n" + "=" * 80)
    print("      DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
