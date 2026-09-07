# evaluation_metrics.py
import numpy as np
import pandas as pd

def correlation_preservation(orig_window: pd.DataFrame, new_window: pd.DataFrame) -> float:
    """
    Frobenius norm of the difference between correlation matrices.
    Lower values indicate better feature correlation preservation.
    """
    corr_orig = orig_window.corr().values
    corr_new = new_window.corr().values
    return np.linalg.norm(corr_orig - corr_new, 'fro')

def proximity(orig_window: pd.DataFrame, new_window: pd.DataFrame) -> float:
    """Euclidean distance between flattened windows."""
    from utils.survival_utils import flatten_window
    return np.linalg.norm(flatten_window(orig_window) - flatten_window(new_window))

def sparsity(explanation: list) -> int:
    """Number of features changed."""
    return len(explanation)

def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets of changed features."""
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)