"""Tests for benchmark dataset loaders and synthetic generator."""

import numpy as np
import pytest
from src.nacc.datasets import generate_synthetic_noisy_data, list_available_datasets, load_dataset


def test_list_datasets():
    datasets = list_available_datasets()
    assert "07_golub_all_aml" in datasets
    assert "01_uci_pancan_rnaseq" in datasets
    assert "synthetic_high_d" in datasets


def test_synthetic_data_generator():
    X, y, meta = generate_synthetic_noisy_data(
        n_samples=150,
        d_signal=5,
        d_noise=95,
        n_clusters=3,
        random_state=42,
    )
    assert X.shape == (150, 100)
    assert len(y) == 150
    assert len(np.unique(y)) == 3
    assert meta["d_signal"] == 5
    assert meta["d_noise"] == 95


def test_load_real_dataset():
    # Load Golub Leukemia dataset
    X, y, meta = load_dataset("07_golub_all_aml")
    assert X.shape[0] == 72
    assert X.shape[1] == 5000
    assert len(y) == 72
    assert meta["n_clusters"] == 2
    assert not np.isnan(X).any()
