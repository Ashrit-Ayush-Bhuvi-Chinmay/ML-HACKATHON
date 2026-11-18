"""Metrics (definitions are ours; the brief leaves most of them open).

Observation level, on reliably labelled points only (labels.py, config.LABEL_MAX_DIST):
  top1        share of points whose highest-scored candidate is the reference arc. The
              denominator includes points whose reference is not in the candidate set and points
              with no candidates at all, so candidate misses count as errors.
Pair level (point x candidate, binary "is this the reference arc"):
  macro_f1    F1 averaged over the two classes, threshold 0.5
  precision / recall of the positive class, log_loss
Route level:
  continuity  share of consecutive distinct predicted arcs that connect head -> tail
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def point_keys(trip_codes: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return trip_codes.astype(np.int64) * 1_000_000 + obs.astype(np.int64)


def best_per_point(keys: np.ndarray, score: np.ndarray):
    """(unique point keys, row index of the highest score for each). Ties -> first row."""
    order = np.lexsort((-score, keys))
    k = keys[order]
    first = np.r_[True, k[1:] != k[:-1]]
    return k[first], order[first]


def pair_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int8)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def top1_accuracy(keys: np.ndarray, y: np.ndarray, score: np.ndarray, n_points: int) -> float:
    _, idx = best_per_point(keys, score)
    return float(y[idx].sum()) / max(n_points, 1)


def route_continuity(arc_seq: np.ndarray, arcs: np.ndarray) -> float:
    """arc_seq: predicted arc per point in time order; arcs: (E, 2) from/to node table."""
    arc_seq = np.asarray(arc_seq)
    if len(arc_seq) < 2:
        return 1.0
    s = arc_seq[np.r_[True, arc_seq[1:] != arc_seq[:-1]]]
    if len(s) < 2:
        return 1.0
    return float(np.mean(arcs[s[:-1], 1] == arcs[s[1:], 0]))
