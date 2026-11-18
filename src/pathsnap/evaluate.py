"""Metrics (definitions are ours; the brief leaves most of them open).

Observation level, on reliably labelled points only (labels.py, config.LABEL_MAX_DIST):
  top1        share of points whose highest-scored candidate is the reference arc. The
              denominator includes points whose reference is not in the candidate set and points
              with no candidates at all, so candidate misses count as errors.
Pair level (point x candidate, binary "is this the reference arc"):
  macro_f1    F1 averaged over the two classes, threshold 0.5
  precision / recall of the positive class, log_loss
Route level:
  strict_adjacency    share of consecutive distinct predicted arcs that share a node exactly
                      (head of one = tail of the next). Pessimistic at 1 Hz: a car often
                      crosses a short arc between two samples.
  plausible continuity (hmm.plausible_continuity) the agreed continuity metric: the next arc
                      is reachable over the network within a distance budget.
  coverage P/R/F1     length-weighted overlap of the predicted and reference arc SETS
                      (order-insensitive: "did we cover the right roads?")
  sequence P/R/F1     length-weighted longest common subsequence of the two arc SEQUENCES
                      (order- and repetition-aware: "did we recover the journey?")
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


def strict_adjacency(arc_seq: np.ndarray, arcs: np.ndarray) -> float | None:
    """arc_seq: predicted arc per point in time order; arcs: (E, 2) from/to node table.
    None when there is no change of arc to judge."""
    arc_seq = np.asarray(arc_seq)
    if len(arc_seq) < 2:
        return None
    s = arc_seq[np.r_[True, arc_seq[1:] != arc_seq[:-1]]]
    if len(s) < 2:
        return None
    return float(np.mean(arcs[s[:-1], 1] == arcs[s[1:], 0]))


def arc_lengths_m(trip, proj, ids) -> dict[int, float]:
    """Length in metres of the given arc ids (projected with the trip's projector)."""
    ids = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64)
    if len(ids) == 0:
        return {}
    u, v = trip.arcs[ids, 0], trip.arcs[ids, 1]
    x1, y1 = proj.to_xy(trip.nodes[u, 0], trip.nodes[u, 1])
    x2, y2 = proj.to_xy(trip.nodes[v, 0], trip.nodes[v, 1])
    return dict(zip(ids.tolist(), np.hypot(x2 - x1, y2 - y1).tolist()))


def weighted_lcs(a, b, length: dict[int, float]) -> float:
    """Length-weighted longest common subsequence of two arc-id sequences, in metres.
    Row-by-row DP with a running prefix maximum: O(len(a) * len(b)) with numpy rows."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    wb = np.array([length[int(x)] for x in b])
    prev = np.zeros(len(b) + 1)
    for x in a:
        cand = np.where(b == x, prev[:-1] + wb, -np.inf)
        prev = np.r_[0.0, np.maximum.accumulate(np.maximum(prev[1:], cand))]
    return float(prev[-1])


def route_scores(pred_seq, ref_seq, length: dict[int, float]) -> dict[str, float]:
    """Coverage (set) and sequence (order-aware) precision / recall / F1, plus the
    Newson-Krumm-style mismatch fraction, all length-weighted."""
    def f1(p, r):
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    pred_set, ref_set = set(int(a) for a in pred_seq), set(int(a) for a in ref_seq)
    L = lambda s: sum(length[a] for a in s)
    inter = L(pred_set & ref_set)
    cov_p = inter / L(pred_set) if L(pred_set) > 0 else 0.0
    cov_r = inter / L(ref_set) if L(ref_set) > 0 else 0.0
    lcs = weighted_lcs(pred_seq, ref_seq, length)
    len_pred = sum(length[int(a)] for a in pred_seq)
    len_ref = sum(length[int(a)] for a in ref_seq)
    seq_p = lcs / len_pred if len_pred > 0 else 0.0
    seq_r = lcs / len_ref if len_ref > 0 else 0.0
    return {
        "coverage_precision": cov_p, "coverage_recall": cov_r, "coverage_f1": f1(cov_p, cov_r),
        "route_mismatch": (L(pred_set - ref_set) + L(ref_set - pred_set)) / L(ref_set) if L(ref_set) > 0 else None,
        "sequence_precision": seq_p, "sequence_recall": seq_r, "sequence_f1": f1(seq_p, seq_r),
    }
