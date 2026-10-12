"""Experiment 4: Downstream Supervised Predictive Utility (PDF Section 9).

Evaluates whether cluster structures provide useful separation for supervised prediction
using KNN, Random Forest, and SVM classifiers.

Run:
    python experiments/exp4_downstream_validation/run_downstream.py
"""

import sys
from pathlib import Path

pkg_root = Path(__file__).resolve().parent.parent.parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

import pandas as pd
from tabulate import tabulate

from src.nacc.datasets import load_dataset
from src.nacc.downstream import evaluate_downstream_utility
from src.nacc.nacc import NoiseAwareConsensusClustering


def main():
    print("=" * 80)
    print("      EXPERIMENT 4: DOWNSTREAM SUPERVISED PREDICTIVE UTILITY")
    print("=" * 80)

    out_dir = Path("experiments") / "exp4_downstream_validation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_datasets = [
        "07_golub_all_aml",
        "06_cll_sub_111",
        "11_geo_gse2034_breast",
    ]

    all_records = []

    for d_key in test_datasets:
        X, y, meta = load_dataset(d_key)
        K = meta["n_clusters"]
        print(f"\n[*] Evaluating Downstream Classifiers on {meta['name']} (N={X.shape[0]}, D={X.shape[1]})...")

        # 1. Baseline raw full-dimensional features (D=5000)
        raw_eval = evaluate_downstream_utility(X, y, cluster_labels=None, active_features=None)

        # 2. Fit NACC to extract clusters and active features
        nacc = NoiseAwareConsensusClustering(
            n_clusters=K,
            n_base_models=15,
            feature_optimizer="ga",
            max_optimization_iter=5,
            random_state=42,
        )
        c_pred = nacc.fit_predict(X)

        # 3. Downstream evaluation on NACC active subspace + cluster structure
        nacc_eval = evaluate_downstream_utility(
            X, y, cluster_labels=c_pred, active_features=nacc.selected_features_
        )

        for clf_name in ["knn", "rf", "svm"]:
            all_records.append({
                "dataset": meta["name"],
                "classifier": clf_name.upper(),
                "raw_accuracy": raw_eval[clf_name]["accuracy"],
                "raw_f1": raw_eval[clf_name]["f1_macro"],
                "nacc_accuracy": nacc_eval[clf_name]["accuracy"],
                "nacc_f1": nacc_eval[clf_name]["f1_macro"],
                "delta_accuracy": round(nacc_eval[clf_name]["accuracy"] - raw_eval[clf_name]["accuracy"], 4),
            })

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "downstream_validation_summary.csv", index=False)

    print("\n" + tabulate(
        df[["dataset", "classifier", "raw_accuracy", "nacc_accuracy", "delta_accuracy", "raw_f1", "nacc_f1"]],
        headers=["Dataset", "Classifier", "Raw Acc", "NACC Acc", "Delta Acc", "Raw F1", "NACC F1"],
        tablefmt="github",
    ))

    print(f"\n[+] Downstream validation results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
