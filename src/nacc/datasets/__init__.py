"""Multi-domain benchmark dataset loaders and synthetic generator.

Provides unified interfaces for high-dimensional microarray, RNA-seq, and mass-spec datasets:
- 01_uci_pancan_rnaseq
- 02_uci_arcene
- 03_gcm
- 04_lung
- 05_carcinom
- 06_cll_sub_111
- 07_golub_all_aml
- 11_geo_gse2034_breast
- synthetic_high_d
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def get_data_dir() -> Path:
    """Returns the path to the benchmark datasets directory."""
    # 1. Environment variable override
    env_dir = os.environ.get("NACC_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    # 2. Look relative to repository / package root
    pkg_dir = Path(__file__).resolve().parent.parent.parent.parent
    clean_dir = pkg_dir / "data" / "clean"
    if clean_dir.exists():
        return clean_dir

    # 3. Fallback to current working directory
    cwd_clean = Path.cwd() / "data" / "clean"
    if cwd_clean.exists():
        return cwd_clean

    # 4. Fallback to user downloads folder if available
    user_download = Path.home() / "Downloads" / "clean" / "clean"
    if user_download.exists():
        return user_download

    return clean_dir


DATASET_REGISTRY = {
    "01_uci_pancan_rnaseq": {
        "file": "01_uci_pancan_rnaseq.csv",
        "name": "Pan-Cancer RNA-Seq",
        "k": 5,
        "d": 5000,
    },
    "02_uci_arcene": {
        "file": "02_uci_arcene.csv",
        "name": "UCI Arcene Mass Spectrometry",
        "k": 2,
        "d": 10000,
    },
    "03_gcm": {
        "file": "03_gcm.csv",
        "name": "Global Cancer Map",
        "k": 14,
        "d": 5000,
    },
    "04_lung": {
        "file": "04_lung.csv",
        "name": "Lung Carcinoma",
        "k": 5,
        "d": 3312,
    },
    "05_carcinom": {
        "file": "05_carcinom.csv",
        "name": "Carcinoma 11-Class",
        "k": 11,
        "d": 5000,
    },
    "06_cll_sub_111": {
        "file": "06_cll_sub_111.csv",
        "name": "CLL Subtypes",
        "k": 3,
        "d": 5000,
    },
    "07_golub_all_aml": {
        "file": "07_golub_all_aml.csv",
        "name": "Golub Leukemia (ALL vs AML)",
        "k": 2,
        "d": 5000,
    },
    "11_geo_gse2034_breast": {
        "file": "11_geo_gse2034_breast.csv",
        "name": "GSE2034 Breast Cancer",
        "k": 2,
        "d": 5000,
    },
}


def list_available_datasets() -> List[str]:
    """Returns a list of all supported dataset keys."""
    return list(DATASET_REGISTRY.keys()) + ["synthetic_high_d"]


def load_dataset(
    name: str,
    data_dir: Optional[Union[str, Path]] = None,
    impute_nan: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[str, int]]]:
    """Loads a benchmark dataset and returns (X, y, meta).

    Args:
        name: Name of the dataset (e.g. '07_golub_all_aml', '01_uci_pancan_rnaseq')
        data_dir: Optional path override for data folder
        impute_nan: Whether to impute missing values (mean imputation)

    Returns:
        X: Feature matrix of shape (N, D)
        y: Numeric integer labels array of shape (N,)
        meta: Dictionary containing metadata (name, n_samples, n_features, n_clusters)
    """
    if name == "synthetic_high_d":
        return generate_synthetic_noisy_data()

    if name not in DATASET_REGISTRY:
        # Search by prefix or substring
        matches = [k for k in DATASET_REGISTRY if name.lower() in k.lower()]
        if matches:
            name = matches[0]
        else:
            raise ValueError(
                f"Dataset '{name}' not found. Available: {list_available_datasets()}"
            )

    info = DATASET_REGISTRY[name]
    folder = Path(data_dir) if data_dir is not None else get_data_dir()
    csv_path = folder / info["file"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Detect sample_id column
    id_col = "sample_id" if "sample_id" in df.columns else None
    label_col = "label" if "label" in df.columns else df.columns[-1]

    # Extract labels
    y_raw = df[label_col].values

    # Drop sample_id and label columns to get feature matrix
    drop_cols = [c for c in [id_col, label_col] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)
    X = feature_df.values.astype(np.float64)

    # Impute missing values if any
    if impute_nan and np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        # In case entire column is NaN, fill with 0
        col_means = np.nan_to_num(col_means, nan=0.0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    # Convert y_raw to clean 0..K-1 integers
    # Note: handle NaNs in labels (e.g. Arcene test samples) by assigning -1
    unique_labels = []
    y = np.full(len(y_raw), -1, dtype=int)
    for i, val in enumerate(y_raw):
        if pd.isna(val) or val == -1.0 or val == "-1" or val == "nan":
            # Check if dataset has -1 as an actual valid class (like Arcene)
            pass

    # Map classes
    non_null_mask = ~pd.isna(y_raw)
    raw_unique = pd.Series(y_raw[non_null_mask]).unique()
    valid_classes = sorted(list(raw_unique), key=lambda x: str(x))

    class_to_idx = {cls: idx for idx, cls in enumerate(valid_classes)}
    for i, val in enumerate(y_raw):
        if not pd.isna(val) and val in class_to_idx:
            y[i] = class_to_idx[val]

    n_clusters = len(valid_classes) if len(valid_classes) > 0 else info["k"]

    meta = {
        "dataset_key": name,
        "name": info["name"],
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_clusters": n_clusters,
        "class_mapping": class_to_idx,
    }

    return X, y, meta


def generate_synthetic_noisy_data(
    n_samples: int = 300,
    d_signal: int = 10,
    d_noise: int = 1990,
    n_clusters: int = 3,
    cluster_std: float = 1.0,
    noise_distribution: str = "gaussian",
    outlier_ratio: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[str, int]]]:
    """Generates synthetic high-dimensional data with controlled signal and distractor dimensions.

    Features [0 ... d_signal - 1] contain distinct cluster Gaussian modes.
    Features [d_signal ... d_total - 1] contain uninformative noise.

    Args:
        n_samples: Total number of observations
        d_signal: Number of true informative dimensions
        d_noise: Number of nuisance noise dimensions
        n_clusters: Number of true clusters
        cluster_std: Standard deviation of signal modes
        noise_distribution: 'gaussian', 'uniform', or 'heavy_tailed'
        outlier_ratio: Fraction of random outlier samples
        random_state: Random seed for reproducibility

    Returns:
        X: Synthetic matrix (N, D)
        y: Integer cluster labels (N,)
        meta: Metadata dictionary
    """
    rng = np.random.default_rng(random_state)
    d_total = d_signal + d_noise

    # Cluster sizes
    base_size = n_samples // n_clusters
    sizes = [base_size] * n_clusters
    sizes[-1] += n_samples - sum(sizes)

    # True cluster centers in signal subspace
    centers = rng.normal(loc=0.0, scale=3.5, size=(n_clusters, d_signal))
    # Ensure separation between centers
    for k in range(n_clusters):
        centers[k] += (k - n_clusters / 2.0) * 2.0

    X_signal_list = []
    y_list = []
    for k, size in enumerate(sizes):
        c_k = centers[k]
        points = rng.normal(loc=c_k, scale=cluster_std, size=(size, d_signal))
        X_signal_list.append(points)
        y_list.extend([k] * size)

    X_signal = np.vstack(X_signal_list)
    y = np.array(y_list, dtype=int)

    # Distractor noise dimensions
    if noise_distribution == "gaussian":
        X_noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, d_noise))
    elif noise_distribution == "uniform":
        X_noise = rng.uniform(low=-2.0, high=2.0, size=(n_samples, d_noise))
    elif noise_distribution == "heavy_tailed":
        X_noise = rng.standard_t(df=3, size=(n_samples, d_noise))
    else:
        X_noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, d_noise))

    # Optional outliers
    n_outliers = int(n_samples * outlier_ratio)
    if n_outliers > 0:
        outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)
        X_signal[outlier_idx] += rng.normal(loc=0.0, scale=5.0, size=(n_outliers, d_signal))

    # Combine signal and noise features
    X = np.hstack([X_signal, X_noise])

    # Shuffle instances
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]

    meta = {
        "dataset_key": "synthetic_high_d",
        "name": f"Synthetic High-D (D={d_total}, K={n_clusters})",
        "n_samples": n_samples,
        "n_features": d_total,
        "d_signal": d_signal,
        "d_noise": d_noise,
        "n_clusters": n_clusters,
        "signal_indices": list(range(d_signal)),
    }

    return X, y, meta
