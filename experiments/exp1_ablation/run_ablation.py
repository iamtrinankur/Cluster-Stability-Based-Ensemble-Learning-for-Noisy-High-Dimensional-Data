"""Experiment 1: Formal 6-Stage Ablation Ladder (PDF Section 10).

Evaluates:
1. Conventional clustering (K-Means)
2. Noise-aware base clustering
3. Noise-aware clustering + GA/PSO feature optimization
4. + Sample stability (Core/Halo modeling)
5. + Error-aware consensus
6. Full NACC (Dissimilarity regularization + reliability profiler)

Run:
    python experiments/exp1_ablation/run_ablation.py
"""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

import pandas as pd
from tabulate import tabulate

from src.nacc.ablation import run_ablation_ladder
from src.nacc.datasets import generate_synthetic_noisy_data, load_dataset


def main():
    print("=" * 75)
    print("      EXPERIMENT 1: FORMAL 6-STAGE ABLATION LADDER (PDF SECTION 10)")
    print("=" * 75)

    out_dir = Path("experiments") / "exp1_ablation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ablation on Synthetic High-Dimensional Noisy Benchmark (D=2,000, 99% noise)
    print("\n[*] Running Ablation Ladder on Synthetic Benchmark (D=2000, 99.8% Noise)...")
    X_syn, y_syn, meta_syn = generate_synthetic_noisy_data(
        n_samples=200, d_signal=4, d_noise=1996, n_clusters=3, random_state=42
    )
    res_syn = run_ablation_ladder(X_syn, y_syn, n_clusters=3, n_base_models=12, random_state=42)

    df_syn = pd.DataFrame(res_syn)
    df_syn.to_csv(out_dir / "ablation_synthetic_results.csv", index=False)

    print("\n" + tabulate(
        df_syn[["stage", "name", "ari", "nmi", "purity", "silhouette", "dunn_index"]],
        headers=["Stage", "Architecture", "ARI", "NMI", "Purity", "Silhouette", "Dunn Index"],
        tablefmt="github",
    ))

    # 2. Ablation on Real Cancer Benchmark: Golub Leukemia (D=5,000, N=72, K=2)
    print("\n[*] Running Ablation Ladder on Golub Leukemia Microarray (D=5000, N=72, K=2)...")
    X_gol, y_gol, meta_gol = load_dataset("07_golub_all_aml")
    res_gol = run_ablation_ladder(X_gol, y_gol, n_clusters=2, n_base_models=12, random_state=42)

    df_gol = pd.DataFrame(res_gol)
    df_gol.to_csv(out_dir / "ablation_golub_results.csv", index=False)

    print("\n" + tabulate(
        df_gol[["stage", "name", "ari", "nmi", "purity", "silhouette", "dunn_index"]],
        headers=["Stage", "Architecture", "ARI", "NMI", "Purity", "Silhouette", "Dunn Index"],
        tablefmt="github",
    ))

    print(f"\n[+] Ablation results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
