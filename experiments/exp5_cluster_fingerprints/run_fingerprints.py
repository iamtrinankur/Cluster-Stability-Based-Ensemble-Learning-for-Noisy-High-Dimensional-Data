"""Experiment 5: Multi-Regime Cluster Reliability Profiling & Fingerprints (PDF Section 8).

Generates multi-dimensional radar fingerprints for clusters across real biomedical datasets.
Distinguishes genuinely robust clusters from fragile/uncertain ones under:
- Noise
- Feature Dropout
- Sample Variation
- Distribution Shift
- Batch Effects

Run:
    python experiments/exp5_cluster_fingerprints/run_fingerprints.py
"""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

import pandas as pd
from tabulate import tabulate

from src.nacc.datasets import load_dataset
from src.nacc.nacc import NoiseAwareConsensusClustering
from src.nacc.viz import plot_reliability_radar


def main():
    print("=" * 80)
    print("      EXPERIMENT 5: CLUSTER RELIABILITY FINGERPRINTS & PROFILES")
    print("=" * 80)

    out_dir = Path("experiments") / "exp5_cluster_fingerprints" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Golub Leukemia Dataset
    X, y, meta = load_dataset("07_golub_all_aml")
    K = meta["n_clusters"]

    print(f"\n[*] Profiling Golub Leukemia Cohort (N={X.shape[0]}, D={X.shape[1]}, K={K})...")
    nacc = NoiseAwareConsensusClustering(
        n_clusters=K,
        n_base_models=15,
        feature_optimizer="ga",
        profiling_repeats=5,
        random_state=42,
    )
    nacc.fit(X)

    table_data = []
    for fp in nacc.fingerprints_:
        table_data.append({
            "dataset": meta["name"],
            "cluster_id": fp.cluster_id,
            "size": fp.size,
            "noise_stability": fp.noise_stability,
            "dropout_stability": fp.dropout_stability,
            "sample_stability": fp.sample_stability,
            "shift_stability": fp.shift_stability,
            "batch_stability": fp.batch_stability,
            "overall_reliability": fp.overall_reliability,
            "grade": fp.stability_grade,
        })

    df = pd.DataFrame(table_data)
    df.to_csv(out_dir / "golub_cluster_fingerprints.csv", index=False)

    print("\n" + tabulate(
        df[["cluster_id", "size", "noise_stability", "dropout_stability", "sample_stability", "shift_stability", "batch_stability", "overall_reliability", "grade"]],
        headers=["Cluster", "Size", "Noise", "Dropout", "Sample", "Shift", "Batch", "Mean Rel", "Grade"],
        tablefmt="github",
    ))

    # Save radar figure
    plot_reliability_radar(
        nacc.fingerprints_,
        save_path=out_dir / "golub_fingerprints_radar.png",
        title="Golub Leukemia: Cluster Reliability Fingerprints",
    )

    print(f"\n[+] Cluster fingerprints saved to: {out_dir / 'golub_fingerprints_radar.png'}")


if __name__ == "__main__":
    main()
