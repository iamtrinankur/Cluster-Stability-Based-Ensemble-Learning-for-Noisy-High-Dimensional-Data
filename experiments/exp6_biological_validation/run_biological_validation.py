"""Experiment 6: Biological Ground-Truth & Oncogenic Biomarker Validation (Plan C).

Validates whether unsupervised feature subsets selected by NACC on high-dimensional
cancer transcriptomics correspond to canonical leukemia and oncogenic driver genes:
- Golub Leukemia Microarray (72 samples x 5,000 genes; AML vs ALL)
- Over-representation hypergeometric test against canonical markers (CST3, MPO, CD33, CD19, CD10, etc.)

Run:
    python experiments/exp6_biological_validation/run_biological_validation.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
pkg_root = Path(__file__).resolve().parent.parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

import pandas as pd
from tabulate import tabulate

from src.nacc.biology import GOLUB_CANONICAL_MARKERS, evaluate_biomarker_enrichment
from src.nacc.datasets import load_dataset
from src.nacc.nacc import NoiseAwareConsensusClustering


def main():
    print("=" * 80)
    print("      EXPERIMENT 6: BIOLOGICAL PATHWAY & BIOMARKER VALIDATION (PLAN C)")
    print("  Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data")
    print("=" * 80)

    out_dir = Path("experiments") / "exp6_biological_validation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Golub Leukemia Dataset
    print("\n[*] Loading Golub Leukemia Dataset (D = 5,000, N = 72, K = 2)...")
    X, y_true, meta = load_dataset("07_golub_all_aml")

    # Read column names directly from CSV to obtain genuine probe IDs
    data_path = Path("data/clean/07_golub_all_aml.csv")
    df_raw = pd.read_csv(data_path, nrows=1)
    feature_names = [c for c in df_raw.columns if c not in ["sample_id", "label"]]

    # 2. Fit NACC Framework
    print("\n[*] Fitting NACC (GA Optimization + Bimodality Screening + Core/Halo Consensus)...")
    nacc = NoiseAwareConsensusClustering(
        n_clusters=2,
        n_base_models=15,
        feature_optimizer="ga",
        max_optimization_iter=8,
        profiling_repeats=3,
        random_state=42,
    )
    y_pred = nacc.fit_predict(X)

    # 3. Analyze Selected Active Subspace Features
    active_idx = nacc.selected_features_
    active_genes = [feature_names[i] for i in active_idx]
    print(f"    -> Active Features Retained: {len(active_idx)} of {len(feature_names)}")

    # 4. Biomarker Over-Representation Analysis
    enrichment = evaluate_biomarker_enrichment(feature_names, active_idx)

    print("\n[*] Biological Enrichment Over-Representation Statistics:")
    print(f"    - Total Genes in Cohort:         {enrichment['total_dataset_features']}")
    print(f"    - Canonical Benchmark Markers:   {enrichment['n_target_biomarkers']}")
    print(f"    - NACC Selected Gene Signature:  {enrichment['n_selected_features']}")
    print(f"    - Direct Overlapping Biomarkers: {enrichment['overlap_count']}")
    print(f"    - Expected Chance Overlap:       {enrichment['expected_overlap']:.3f}")
    print(f"    - Fold Over-Representation:      {enrichment['fold_enrichment']:.2f}x")
    print(f"    - Hypergeometric p-value:        {enrichment['hypergeometric_p_value']:.4e}")
    print(f"    - Statistically Significant:     {enrichment['is_statistically_significant']}")

    print("\n[+] Recovered Canonical Leukemia Diagnostic Genes:")
    for gene in enrichment["recovered_biomarkers"]:
        print(f"    * {gene}")

    # Save summary
    res_df = pd.DataFrame([{
        "dataset": meta["name"],
        "n_samples": meta["n_samples"],
        "total_features": enrichment["total_dataset_features"],
        "active_features": enrichment["n_selected_features"],
        "overlap_count": enrichment["overlap_count"],
        "fold_enrichment": enrichment["fold_enrichment"],
        "p_value": enrichment["hypergeometric_p_value"],
        "statistically_significant": enrichment["is_statistically_significant"],
    }])
    res_df.to_csv(out_dir / "biological_enrichment_summary.csv", index=False)
    print(f"\n[+] Biological validation results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
