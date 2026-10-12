"""Publication-Quality Scientific Visualization Module.

Generates charts and figures:
- Consensus Co-association Matrix Heatmap
- Cluster Reliability Fingerprints (Radar Charts)
- Core / Halo / Uncertain Sample Stability Projections
- Noise-Response Curves (Robustness Experiment)
- Ablation Progression Ladder Chart
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.nacc.reliability import ClusterFingerprint


def plot_consensus_heatmap(
    consensus_matrix: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "NACC Error-Aware Consensus Co-Association Matrix",
) -> plt.Figure:
    """Plots and saves sorted block-diagonal consensus matrix heatmap."""
    N = len(labels)
    # Sort samples by assigned cluster label
    sorted_idx = np.argsort(labels)
    sorted_matrix = consensus_matrix[np.ix_(sorted_idx, sorted_idx)]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(sorted_matrix, cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Consensus Affinity $H_{ij}$")

    # Draw cluster boundary lines
    unique_labels, counts = np.unique(labels, return_counts=True)
    cum_counts = np.cumsum(counts)
    for boundary in cum_counts[:-1]:
        ax.axhline(boundary - 0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(boundary - 0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Reordered Samples ($N$)", fontsize=10)
    ax.set_ylabel("Reordered Samples ($N$)", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_reliability_radar(
    fingerprints: List[ClusterFingerprint],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Cluster Reliability Fingerprints",
) -> plt.Figure:
    """Plots multi-dimensional radar chart of cluster stability across 5 perturbation regimes."""
    categories = ["Noise", "Dropout", "Subsampling", "Distribution Shift", "Batch Effects"]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), dpi=200)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fingerprints), 3)))

    for i, fp in enumerate(fingerprints):
        values = fp.to_vector().tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            color=colors[i],
            linewidth=2,
            label=f"Cluster {fp.cluster_id} ({fp.stability_grade}, Rel={fp.overall_reliability:.2f})",
        )
        ax.fill(angles, values, color=colors[i], alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10, fontweight="medium")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_sample_stability_projection(
    X: np.ndarray,
    categories: np.ndarray,
    stability_scores: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Sample Stability & Core/Halo/Uncertain Partitioning",
) -> plt.Figure:
    """Projects samples onto 2D PCA space colored by Core/Halo/Uncertain status."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

    # 1. Categorical scatter
    color_map = {"Core": "#2ca02c", "Halo": "#ff7f0e", "Uncertain": "#d62728"}
    marker_map = {"Core": "o", "Halo": "^", "Uncertain": "x"}

    for cat in ["Core", "Halo", "Uncertain"]:
        mask = (categories == cat)
        if np.any(mask):
            ax1.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=color_map[cat],
                marker=marker_map[cat],
                s=40 if cat != "Uncertain" else 55,
                alpha=0.8,
                label=f"{cat} (n={np.sum(mask)})",
                edgecolors="none" if cat == "Uncertain" else "black",
                linewidths=0.5,
            )

    ax1.set_title("Categorical Partitioning", fontsize=11, fontweight="bold")
    ax1.set_xlabel(f"PCA-1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PCA-2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Continuous stability score scatter
    sc = ax2.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=stability_scores,
        cmap="viridis",
        s=40,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax2, label="Sample Stability Score $s_i$")
    ax2.set_title("Continuous Stability Spectrum", fontsize=11, fontweight="bold")
    ax2.set_xlabel(f"PCA-1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PCA-2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_noise_response_curves(
    noise_levels: List[float],
    model_curves: Dict[str, List[float]],
    metric_name: str = "Adjusted Rand Index (ARI)",
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Noise-Response Curves: Clustering Degradation vs Noise Level",
) -> plt.Figure:
    """Plots metric degradation curves across increasing perturbation noise."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    styles = {
        "NACC (Proposed)": {"color": "#1f77b4", "linewidth": 2.5, "marker": "o"},
        "Standard Consensus": {"color": "#ff7f0e", "linewidth": 1.8, "marker": "s", "linestyle": "--"},
        "Core-Halo": {"color": "#2ca02c", "linewidth": 1.8, "marker": "^", "linestyle": "--"},
        "Spectral": {"color": "#9467bd", "linewidth": 1.5, "marker": "d", "linestyle": ":"},
        "K-Means": {"color": "#d62728", "linewidth": 1.5, "marker": "v", "linestyle": ":"},
    }

    for model_name, scores in model_curves.items():
        st = styles.get(model_name, {"color": "gray", "linewidth": 1.5, "marker": "."})
        ax.plot(noise_levels, scores, label=model_name, **st)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(r"Additive Gaussian Noise Scale ($\sigma$)", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig
