"""Experiment 2: Real-World Multi-Domain Biomedical Benchmarks.

Evaluates NACC against 6 baseline clustering algorithms across high-dimensional datasets:
- 07_golub_all_aml (72 x 5,000)
- 06_cll_sub_111 (111 x 5,000)
- 04_lung (203 x 3,312)
- 05_carcinom (174 x 5,000)
- 11_geo_gse2034_breast (286 x 5,000)
- 01_uci_pancan_rnaseq (801 x 5,000)

Run:
    python experiments/exp2_real_world_benchmarks/run_real_benchmarks.py
"""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from typing import Dict, List, Union
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.nacc.baselines.classical import run_fuzzy_cmeans, run_kmeans, run_spectral, run_ward
from src.nacc.baselines.consensus import CoreHaloConsensusClustering, StandardConsensusClustering
from src.nacc.datasets import load_dataset
from src.nacc.metrics import evaluate_all
from src.nacc.nacc import NoiseAwareConsensusClustering


def evaluate_cohort(dataset_key: str, max_samples: int = 400) -> List[Dict[str, Union[str, float]]]:
    X, y, meta = load_dataset(dataset_key)
    N, D = X.shape
    K = meta["n_clusters"]

    if N > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        N = len(y)

    print(f"\n[*] Evaluating Dataset: {meta['name']} (N={N}, D={D}, K={K})...")

    models = {
        "K-Means": run_kmeans(X, K, random_state=42),
        "Ward": run_ward(X, K),
        "Spectral": run_spectral(X, K, random_state=42),
        "FCM": run_fuzzy_cmeans(X, K, random_state=42),
        "Standard Consensus": StandardConsensusClustering(n_clusters=K, n_base_models=15, random_state=42).fit_predict(X),
        "Core-Halo": CoreHaloConsensusClustering(n_clusters=K, n_base_models=15, random_state=42).fit_predict(X),
    }

    # NACC model
    nacc = NoiseAwareConsensusClustering(
        n_clusters=K,
        n_base_models=15,
        feature_optimizer="ga",
        subspace_ratio=0.25,
        max_optimization_iter=6,
        profiling_repeats=2,
        random_state=42,
    )
    models["NACC (Proposed)"] = nacc.fit_predict(X)

    records = []
    for m_name, preds in models.items():
        ev = evaluate_all(y, preds, X)
        records.append({
            "dataset": meta["name"],
            "dataset_key": dataset_key,
            "samples": N,
            "features": D,
            "k": K,
            "model": m_name,
            "ari": ev.get("ari", 0.0),
            "nmi": ev.get("nmi", 0.0),
            "purity": ev.get("purity", 0.0),
            "silhouette": ev.get("silhouette", 0.0),
            "dunn_index": ev.get("dunn_index", 0.0),
        })

    return records


def main():
    print("=" * 80)
    print("      EXPERIMENT 2: REAL-WORLD BIOMEDICAL CLUSTERING BENCHMARKS")
    print("=" * 80)

    out_dir = Path("experiments") / "exp2_real_world_benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_cohorts = [
        "07_golub_all_aml",
        "06_cll_sub_111",
        "04_lung",
        "11_geo_gse2034_breast",
    ]

    all_records = []
    for cohort in benchmark_cohorts:
        records = evaluate_cohort(cohort)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "real_world_benchmarks_summary.csv", index=False)

    print("\n" + tabulate(
        df[["dataset", "model", "ari", "nmi", "purity", "silhouette", "dunn_index"]],
        headers=["Dataset", "Model", "ARI", "NMI", "Purity", "Silhouette", "Dunn"],
        tablefmt="github",
    ))

    print(f"\n[+] Benchmark records saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
