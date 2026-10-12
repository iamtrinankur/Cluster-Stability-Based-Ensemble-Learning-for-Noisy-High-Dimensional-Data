import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
#!/usr/bin/env python3
"""Unified Benchmark and Experiment Suite CLI Runner.

Executes all empirical experiments:
- Exp 1: 6-Stage Ablation Ladder
- Exp 2: Real-World Biomedical Cancer Benchmarks
- Exp 3: Noise-Response Degradation Curves
- Exp 4: Downstream Supervised Predictive Utility (KNN, RF, SVM)
- Exp 5: Multi-Regime Cluster Reliability Fingerprints

Usage:
    python scripts/run_benchmarks.py --all
    python scripts/run_benchmarks.py --exp 1
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd_list, description):
    print("\n" + "=" * 75)
    print(f"[*] Running: {description}")
    print("=" * 75)
    result = subprocess.run(cmd_list, text=True)
    if result.returncode != 0:
        print(f"[!] Warning: Step failed with return code {result.returncode}")
    else:
        print(f"[+] Completed: {description}")


def main():
    parser = argparse.ArgumentParser(description="NACC Benchmark & Experiment Runner")
    parser.add_argument("--all", action="store_true", help="Run all 6 empirical experiments")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5, 6], help="Run a specific experiment (1-6)")
    args = parser.parse_args()

    py = sys.executable

    exp_map = {
        1: ([py, "experiments/exp1_ablation/run_ablation.py"], "Exp 1: Formal 6-Stage Ablation Ladder"),
        2: ([py, "experiments/exp2_real_world_benchmarks/run_real_benchmarks.py"], "Exp 2: Real-World Biomedical Benchmarks"),
        3: ([py, "experiments/exp3_noise_robustness/run_noise_experiment.py"], "Exp 3: Noise-Response Curves"),
        4: ([py, "experiments/exp4_downstream_validation/run_downstream.py"], "Exp 4: Downstream Classification Utility"),
        5: ([py, "experiments/exp5_cluster_fingerprints/run_fingerprints.py"], "Exp 5: Cluster Reliability Fingerprints"),
        6: ([py, "experiments/exp6_biological_validation/run_biological_validation.py"], "Exp 6: Plan C Biological Pathway & Biomarker Enrichment"),
    }

    if args.exp:
        cmd, desc = exp_map[args.exp]
        run_cmd(cmd, desc)
    else:
        print("[*] Running all empirical experiment suites...")
        for exp_id in sorted(exp_map.keys()):
            cmd, desc = exp_map[exp_id]
            run_cmd(cmd, desc)


if __name__ == "__main__":
    main()
