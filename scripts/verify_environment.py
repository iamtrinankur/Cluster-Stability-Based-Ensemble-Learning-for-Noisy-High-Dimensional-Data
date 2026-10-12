#!/usr/bin/env python3
"""Environment and Dataset Sanity Verifier for NACC Research Suite."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

def main():
    print("=" * 70)
    print("       NACC Environment & Dataset Verification Suite")
    print("=" * 70)

    # 1. Python version
    py_ver = sys.version_info
    print(f"[+] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    # 2. Required packages
    packages = ["numpy", "scipy", "sklearn", "pandas", "matplotlib", "tabulate", "yaml"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"[+] Package {pkg:<12}: OK")
        except ImportError:
            print(f"[-] Package {pkg:<12}: MISSING")

    # 3. Check datasets
    print("\n--- Benchmark Datasets ---")
    data_dir = _repo_root / "data" / "clean"
    expected = [
        "01_uci_pancan_rnaseq.csv",
        "02_uci_arcene.csv",
        "03_gcm.csv",
        "04_lung.csv",
        "05_carcinom.csv",
        "06_cll_sub_111.csv",
        "07_golub_all_aml.csv",
        "11_geo_gse2034_breast.csv",
        "SUMMARY.csv",
    ]
    found = 0
    for name in expected:
        p = data_dir / name
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [FOUND] {name:<30} ({size_mb:>6.2f} MB)")
            found += 1
        else:
            print(f"  [MISSING] {name}")
    print(f"Datasets: {found}/{len(expected)} verified.")

    # 4. Pipeline quick sanity test
    print("\n--- Pipeline Sanity Test ---")
    try:
        from src.nacc.datasets import generate_synthetic_noisy_data
        from src.nacc.nacc import NoiseAwareConsensusClustering
        X, y, meta = generate_synthetic_noisy_data(n_samples=50, d_signal=5, d_noise=45, n_clusters=2)
        model = NoiseAwareConsensusClustering(n_clusters=2, n_base_models=5, feature_optimizer="none", profiling_repeats=1)
        labels = model.fit_predict(X)
        print(f"[+] Synthetic sanity test successful (Clusters: {sorted(list(set(labels)))}).")
    except Exception as e:
        print(f"[-] Sanity test failed: {e}")

    print("=" * 70)
    print("Verification completed.")
    print("=" * 70)

if __name__ == "__main__":
    main()

