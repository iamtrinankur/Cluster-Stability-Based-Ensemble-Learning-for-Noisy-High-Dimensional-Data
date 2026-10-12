"""Command-Line Interface (CLI) for Noise-Aware Consensus Clustering (NACC).

Topic: Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data

Usage:
    # Run on built-in benchmark dataset
    python -m src.nacc.cli --dataset 07_golub_all_aml --n-clusters 2 --out-dir results/

    # Run on custom high-dimensional CSV
    python -m src.nacc.cli --data path/to/data.csv --n-clusters 3 --label-col label --out-dir results/
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.nacc.datasets import load_dataset
from src.nacc.metrics import evaluate_all
from src.nacc.nacc import NoiseAwareConsensusClustering
from src.nacc.viz import plot_consensus_heatmap, plot_reliability_radar, plot_sample_stability_projection


def parse_args():
    parser = argparse.ArgumentParser(
        description="NACC: Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        type=str,
        help="Built-in benchmark dataset key (e.g. 07_golub_all_aml, 04_lung, 06_cll_sub_111, 03_gcm)",
    )
    group.add_argument(
        "--data",
        type=str,
        help="Path to custom CSV dataset file",
    )

    parser.add_argument("--n-clusters", type=int, default=None, help="Number of clusters K (required for custom CSV)")
    parser.add_argument("--label-col", type=str, default="label", help="Name of ground-truth label column in custom CSV")
    parser.add_argument("--sample-id-col", type=str, default="sample_id", help="Name of sample ID column in custom CSV")
    parser.add_argument("--n-base-models", type=int, default=15, help="Number of base clustering ensemble views")
    parser.add_argument("--optimizer", type=str, choices=["ga", "pso", "none"], default="ga", help="Feature optimizer method")
    parser.add_argument("--max-iter", type=int, default=6, help="Maximum feature optimization iterations")
    parser.add_argument("--profiling-repeats", type=int, default=3, help="Repeats for multi-factor reliability profiling")
    parser.add_argument("--report", action="store_true", help="Generate interactive HTML report dashboard")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory to save output files and plots")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      NOISE-AWARE CONSENSUS CLUSTERING (NACC) - CLI RUNNER")
    print("  Cluster Stability-Based Ensemble Learning for Noisy High-Dimensional Data")
    print("=" * 80)

    # 1. Load Data
    if args.dataset:
        print(f"\n[*] Loading built-in benchmark dataset: '{args.dataset}'...")
        X, y_true, meta = load_dataset(args.dataset)
        K = args.n_clusters if args.n_clusters is not None else meta["n_clusters"]
        dataset_name = meta["name"]
        sample_ids = [f"Sample_{i}" for i in range(len(X))]
        csv_file = Path("data/clean") / f"{args.dataset}.csv"
        if csv_file.exists():
            df_head = pd.read_csv(csv_file, nrows=1)
            feature_names = [c for c in df_head.columns if c not in ["sample_id", "label"]]
        else:
            feature_names = [f"Feature_{j}" for j in range(X.shape[1])]
    else:
        csv_path = Path(args.data)
        if not csv_path.exists():
            print(f"[!] Error: File not found: {csv_path}")
            sys.exit(1)
        print(f"\n[*] Reading custom CSV data from: '{csv_path}'...")
        df = pd.read_csv(csv_path)

        sample_ids = df[args.sample_id_col].tolist() if args.sample_id_col in df.columns else [f"Sample_{i}" for i in range(len(df))]
        y_true = df[args.label_col].values if args.label_col in df.columns else None

        drop_cols = [c for c in [args.sample_id_col, args.label_col] if c in df.columns]
        feature_df = df.drop(columns=drop_cols)
        feature_names = feature_df.columns.tolist()
        X = feature_df.values.astype(np.float64)

        if args.n_clusters is None:
            if y_true is not None:
                K = len(np.unique(y_true))
            else:
                print("[!] Error: Must specify --n-clusters when running on unlabelled custom data.")
                sys.exit(1)
        else:
            K = args.n_clusters
        dataset_name = csv_path.stem

    N, D = X.shape
    print(f"    - Cohort Name:       {dataset_name}")
    print(f"    - Sample Count (N):  {N}")
    print(f"    - Feature Count (D): {D}")
    print(f"    - Target Clusters (K): {K}")

    # 2. Fit NACC Estimator
    print(f"\n[*] Initializing NACC with optimizer='{args.optimizer}', views={args.n_base_models}...")
    t0 = time.time()
    nacc = NoiseAwareConsensusClustering(
        n_clusters=K,
        n_base_models=args.n_base_models,
        feature_optimizer=args.optimizer if args.optimizer != "none" else None,
        max_optimization_iter=args.max_iter,
        profiling_repeats=args.profiling_repeats,
        random_state=args.random_state,
    )
    y_pred = nacc.fit_predict(X)
    elapsed = time.time() - t0
    print(f"[+] Fitting completed in {elapsed:.2f} seconds.")

    # 3. Compile Partition & Sample Stability Data
    stab = nacc.stability_profile_
    results_df = pd.DataFrame({
        "sample_id": sample_ids,
        "predicted_cluster": y_pred,
        "stability_score": np.round(stab.stability_scores, 4),
        "category": [
            "Core" if stab.core_mask[i] else ("Halo" if stab.halo_mask[i] else "Uncertain")
            for i in range(N)
        ],
    })
    if y_true is not None:
        results_df["ground_truth"] = y_true

    results_csv = out_dir / f"{dataset_name}_predictions.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n[+] Cluster predictions & sample stability saved to: {results_csv}")

    # 4. Display Core/Halo Stratification
    n_core = np.sum(stab.core_mask)
    n_halo = np.sum(stab.halo_mask)
    n_unc = np.sum(stab.uncertain_mask)
    print("\n[*] Sample Stability Stratification:")
    print(f"    - Core (Rock-Solid):       {n_core} / {N} ({n_core / N * 100:.1f}%)")
    print(f"    - Halo (Boundary):         {n_halo} / {N} ({n_halo / N * 100:.1f}%)")
    print(f"    - Uncertain (Noise-Prone): {n_unc} / {N} ({n_unc / N * 100:.1f}%)")

    # Save 2D sample stability projection plot
    cat_arr = np.array(results_df["category"].values)
    proj_path = out_dir / f"{dataset_name}_sample_stability.png"
    plot_sample_stability_projection(X, cat_arr, stab.stability_scores, save_path=proj_path, title=f"{dataset_name}: Sample Stability")
    print(f"[+] Saved sample stability projection to: {proj_path}")

    # 5. Validation Metrics
    if y_true is not None:
        metrics = evaluate_all(y_true, y_pred, X)
        print("\n[*] Clustering Validation Metrics:")
        metrics_table = [
            ["Adjusted Rand Index (ARI)", f"{metrics['ari']:.4f}"],
            ["Normalized Mutual Info (NMI)", f"{metrics['nmi']:.4f}"],
            ["Cluster Purity", f"{metrics['purity']:.4f}"],
            ["Silhouette Coefficient", f"{metrics['silhouette']:.4f}"],
            ["Dunn Separation Index", f"{metrics['dunn_index']:.4f}"],
            ["Davies-Bouldin Index", f"{metrics['davies_bouldin']:.4f}"],
            ["Calinski-Harabasz Score", f"{metrics['calinski_harabasz']:.2f}"],
        ]
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="github"))

    # 6. Cluster Reliability Fingerprints
    if nacc.fingerprints_:
        print("\n[*] Cluster Reliability Fingerprints (Multi-Regime Stress Testing):")
        fp_rows = []
        for fp in nacc.fingerprints_:
            fp_rows.append([
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
            fp_rows,
            headers=["Cluster", "Size", "Noise", "Dropout", "Sample", "Shift", "Batch", "Mean Rel", "Grade"],
            tablefmt="github",
        ))

        # Save radar plot
        radar_path = out_dir / f"{dataset_name}_reliability_radar.png"
        plot_reliability_radar(nacc.fingerprints_, save_path=radar_path, title=f"{dataset_name}: Cluster Reliability")
        print(f"[+] Saved reliability radar chart to: {radar_path}")

    # 7. Consensus Heatmap
    if nacc.consensus_matrix_ is not None:
        heatmap_path = out_dir / f"{dataset_name}_consensus_matrix.png"
        plot_consensus_heatmap(nacc.consensus_matrix_, y_pred, save_path=heatmap_path, title=f"{dataset_name}: Consensus Affinity")
        print(f"[+] Saved consensus matrix heatmap to: {heatmap_path}")

    # 8. Interactive HTML Report Generation
    if args.report:
        from src.nacc.report import generate_html_report
        from src.nacc.biology import evaluate_biomarker_enrichment

        bio_data = None
        if args.dataset == "07_golub_all_aml":
            bio_data = evaluate_biomarker_enrichment(feature_names, nacc.selected_features_)

        fp_data = [
            {
                "cluster_id": fp.cluster_id,
                "size": fp.size,
                "noise_stability": fp.noise_stability,
                "dropout_stability": fp.dropout_stability,
                "sample_stability": fp.sample_stability,
                "shift_stability": fp.shift_stability,
                "batch_stability": fp.batch_stability,
                "overall_reliability": fp.overall_reliability,
                "grade": fp.stability_grade,
            }
            for fp in nacc.fingerprints_
        ] if nacc.fingerprints_ else []

        fig_dict = {
            "sample_stability": out_dir / f"{dataset_name}_sample_stability.png",
            "consensus_matrix": out_dir / f"{dataset_name}_consensus_matrix.png",
            "reliability_radar": out_dir / f"{dataset_name}_reliability_radar.png",
        }

        report_file = out_dir / f"{dataset_name}_report.html"
        generate_html_report(
            dataset_name=dataset_name,
            n_samples=N,
            n_features=D,
            n_clusters=K,
            metrics=metrics if y_true is not None else {},
            core_count=int(n_core),
            halo_count=int(n_halo),
            uncertain_count=int(n_unc),
            fingerprints_data=fp_data,
            biomarkers_data=bio_data,
            figure_paths=fig_dict,
            output_path=report_file,
        )
        print(f"\n[+] Interactive scientific HTML report generated: {report_file.resolve()}")

    print("\n" + "=" * 80)
    print("      NACC PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
