"""Evaluation metrics for face verification and identification."""

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        emb1: 1-D numpy array.
        emb2: 1-D numpy array of the same length.

    Returns:
        Cosine similarity in [-1, 1].
    """
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    dot = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    if norm == 0.0:
        return 0.0
    return float(dot / norm)


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute Euclidean distance between two embedding vectors.

    Args:
        emb1: 1-D numpy array.
        emb2: 1-D numpy array of the same length.

    Returns:
        L2 distance (non-negative float).
    """
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return float(np.linalg.norm(emb1 - emb2))


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under the ROC Curve.

    Args:
        labels: Binary ground-truth labels (0 or 1).
        scores: Predicted similarity scores or probabilities.

    Returns:
        AUC value in [0, 1].
    """
    return float(roc_auc_score(labels, scores))


def compute_roc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Receiver Operating Characteristic curve.

    Args:
        labels: Binary ground-truth labels (0 or 1).
        scores: Predicted similarity scores or probabilities.

    Returns:
        Tuple of (fpr, tpr, thresholds).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def find_best_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """Find the threshold that maximises TPR - FPR (Youden's J statistic).

    Args:
        fpr: False positive rates from :func:`compute_roc`.
        tpr: True positive rates from :func:`compute_roc`.
        thresholds: Corresponding thresholds.

    Returns:
        Optimal threshold value.
    """
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])
