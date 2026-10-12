"""Biological Pathway and Biomarker Enrichment Validation.

Implements Plan C: Biological Ground-Truth Validation for High-Dimensional Cancer Cohorts
Validates whether unsupervised feature subsets selected by NACC capture canonical
oncogenic driver genes and diagnostic biomarkers:
- Golub Leukemia (AML vs ALL diagnostic markers: CST3, MPO, CD33, CD19, CD10, etc.)
- TCGA Pan-Cancer RNA-Seq (BRCA, KIRC, LUAD, PRAD, COAD lineage-specific markers)
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import hypergeom


# Canonical Leukemia Diagnostic Biomarkers (Golub et al., Science 1999)
GOLUB_CANONICAL_MARKERS = {
    "AML": [
        "M27891",  # CST3 (Cystatin C) - Top AML marker
        "M23197",  # CD33 antigen (differentiation antigen)
        "X59417",  # MPO (Myeloperoxidase)
        "M16038",  # LYN (tyrosine-protein kinase)
        "M84526",  # DF (D-factor / Adipsin)
        "M22960",  # PPGB (protective protein for beta-galactosidase)
        "M55150",  # FAH (fumarylacetoacetate)
        "U50136",  # LTC4S (leukotriene C4 synthase)
        "X17042",  # PRG1 (proteoglycan 1)
        "Y00787",  # Interleukin 8
    ],
    "ALL": [
        "M28170",  # CD19 antigen (B-cell lineage)
        "J03779",  # MME / CD10 (Common acute lymphocytic leukemia antigen CALLA)
        "M31523",  # TCF3 / E2A (Transcription factor 3)
        "M63138",  # CTSD (Cathepsin D)
        "M13792",  # ADA (Adenosine deaminase)
        "M83652",  # PFC (Properdin P factor)
        "X15949",  # ABL1 (Tyrosine-protein kinase ABL1)
        "U22376",  # MYB (proto-oncogene)
        "X59871",  # TCF7 (transcription factor 7)
    ],
}


def evaluate_biomarker_enrichment(
    feature_names: List[str],
    selected_feature_indices: np.ndarray,
    target_markers: Optional[List[str]] = None,
) -> Dict[str, Union[float, int, List[str]]]:
    """Evaluates hypergeometric over-representation of canonical disease biomarkers.

    Args:
        feature_names: Full list of feature names/probe IDs in dataset (D,)
        selected_feature_indices: Indices of active features selected by NACC
        target_markers: Optional list of probe IDs/genes to test (defaults to Golub markers)

    Returns:
        Dictionary containing overlap count, hypergeometric p-value, and recovered genes.
    """
    total_genes = len(feature_names)
    selected_count = len(selected_feature_indices)

    if target_markers is None:
        target_markers = GOLUB_CANONICAL_MARKERS["AML"] + GOLUB_CANONICAL_MARKERS["ALL"]

    # Normalize probe IDs
    def clean_id(name: str) -> str:
        return name.split("_")[0].split(".")[0].strip().upper()

    cleaned_features = [clean_id(f) for f in feature_names]
    cleaned_targets = set(clean_id(t) for t in target_markers)

    # Find total target markers present in the dataset
    in_dataset_targets = set()
    for idx, f in enumerate(cleaned_features):
        if f in cleaned_targets or any(t in f for t in cleaned_targets):
            in_dataset_targets.add(idx)

    M_total = total_genes
    n_targets = len(in_dataset_targets)
    N_selected = selected_count

    # Find overlap
    selected_set = set(selected_feature_indices)
    overlap_indices = selected_set.intersection(in_dataset_targets)
    k_overlap = len(overlap_indices)

    # Hypergeometric test: P(X >= k)
    if n_targets > 0 and N_selected > 0:
        # hypergeom.sf(k - 1, M_total, n_targets, N_selected)
        p_val = float(hypergeom.sf(k_overlap - 1, M_total, n_targets, N_selected))
        expected_overlap = (n_targets * N_selected) / M_total
        fold_enrichment = float(k_overlap / (expected_overlap + 1e-12))
    else:
        p_val = 1.0
        expected_overlap = 0.0
        fold_enrichment = 0.0

    recovered_names = [feature_names[i] for i in sorted(overlap_indices)]

    return {
        "total_dataset_features": M_total,
        "n_target_biomarkers": n_targets,
        "n_selected_features": N_selected,
        "overlap_count": k_overlap,
        "expected_overlap": round(expected_overlap, 3),
        "fold_enrichment": round(fold_enrichment, 2),
        "hypergeometric_p_value": float(p_val),
        "is_statistically_significant": bool(p_val < 0.05),
        "recovered_biomarkers": recovered_names,
    }
