"""Experiment 3: Controlled Noise and Perturbation Robustness Evaluation.

Implements Sections 8 & 9 of the NACC framework:
Evaluates clustering degradation across increasing levels of additive noise (sigma in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).
Records Noise-Response Curves comparing NACC vs Baselines.

Run:
    python experiments/exp3_noise_robustness/run_noise_experiment.py
"""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.nacc.baselines.classical import run_kmeans, run_spectral
from src.nacc.baselines.consensus import CoreHaloConsensusClustering, StandardConsensusClustering
from src.nacc.datasets import generate_synthetic_noisy_data
from src.nacc.metrics import evaluate_external
from src.nacc.nacc import NoiseAwareConsensusClustering
from src.nacc.viz import plot_noise_response_curves


def main():
    print("=" * 80)
    print("      EXPERIMENT 3: NOISE-RESPONSE CURVES & PERTURBATION SENSITIVITY")
    print("=" * 80)

    out_dir = Path("experiments") / "exp3_noise_robustness" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base cohort
    N = 180
    D_sig = 4
    D_noise = 996
    K = 3
    X_clean, y_true, _ = generate_synthetic_noisy_data(
        n_samples=N, d_signal=D_sig, d_noise=D_noise, n_clusters=K, cluster_std=0.8, random_state=42
    )

    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    model_names = [
        "NACC (Proposed)",
        "Standard Consensus",
        "Core-Halo",
        "Spectral",
        "K-Means",
    ]

    ari_curves = {m: [] for m in model_names}
    records = []

    rng = np.random.default_rng(42)

    for sigma in noise_levels:
        print(f"[*] Testing Noise Scale sigma = {sigma:.1f}...")
        noise = rng.normal(0.0, sigma, size=X_clean.shape) if sigma > 0 else 0.0
        X_pert = X_clean + noise

        # 1. K-Means
        pred_km = run_kmeans(X_pert, K, random_state=42)
        ari_km = evaluate_external(y_true, pred_km)["ari"]
        ari_curves["K-Means"].append(ari_km)

        # 2. Spectral
        pred_spec = run_spectral(X_pert, K, random_state=42)
        ari_spec = evaluate_external(y_true, pred_spec)["ari"]
        ari_curves["Spectral"].append(ari_spec)

        # 3. Standard Consensus
        pred_sc = StandardConsensusClustering(n_clusters=K, n_base_models=12, random_state=42).fit_predict(X_pert)
        ari_sc = evaluate_external(y_true, pred_sc)["ari"]
        ari_curves["Standard Consensus"].append(ari_sc)

        # 4. Core-Halo
        pred_ch = CoreHaloConsensusClustering(n_clusters=K, n_base_models=12, random_state=42).fit_predict(X_pert)
        ari_ch = evaluate_external(y_true, pred_ch)["ari"]
        ari_curves["Core-Halo"].append(ari_ch)

        # 5. NACC
        nacc = NoiseAwareConsensusClustering(
            n_clusters=K,
            n_base_models=12,
            feature_optimizer="ga",
            max_optimization_iter=5,
            profiling_repeats=0,
            random_state=42,
        )
        pred_nacc = nacc.fit_predict(X_pert)
        ari_nacc = evaluate_external(y_true, pred_nacc)["ari"]
        ari_curves["NACC (Proposed)"].append(ari_nacc)

        for m_name, score in [
            ("K-Means", ari_km),
            ("Spectral", ari_spec),
            ("Standard Consensus", ari_sc),
            ("Core-Halo", ari_ch),
            ("NACC (Proposed)", ari_nacc),
        ]:
            records.append({"noise_sigma": sigma, "model": m_name, "ari": score})

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "noise_response_summary.csv", index=False)

    # Plot and save curves
    plot_noise_response_curves(
        noise_levels=noise_levels,
        model_curves=ari_curves,
        metric_name="Adjusted Rand Index (ARI)",
        save_path=out_dir / "noise_response_curves.png",
        title="Noise Degradation: NACC vs Baselines at D=1000",
    )

    # Pivot display table
    pivot_df = df.pivot(index="model", columns="noise_sigma", values="ari")
    print("\n" + tabulate(pivot_df, headers="keys", tablefmt="github"))
    print(f"\n[+] Noise experiment curves saved to: {out_dir / 'noise_response_curves.png'}")


if __name__ == "__main__":
    main()
