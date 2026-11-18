"""Frozen-model route inference for the API and final evaluator prediction stage."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .decoding import adjacency_log_probabilities, viterbi
from .features import candidate_features


def predict_route(observations: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]],
                  model: Any, feature_columns: Sequence[str], radius_m: float = 100,
                  max_candidates: int = 8) -> dict[str, Any]:
    """Predict classifier-only and HMM/Viterbi routes without accessing any route labels."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas before running model inference.") from exc
    candidates = candidate_features(observations, edges, radius_m, max_candidates)
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_key[(str(candidate["trajectory_id"]), int(candidate["point_index"]))].append(candidate)
    ordered_points = sorted(observations, key=lambda p: (str(p["trajectory_id"]), int(p["point_index"])))
    if any(not by_key[(str(p["trajectory_id"]), int(p["point_index"]))] for p in ordered_points):
        raise ValueError("At least one observation has no road candidates within the configured radius.")
    rows: list[dict] = []
    probabilities_by_point: list[list[float]] = []
    candidate_group: list[list[dict]] = []
    for point in ordered_points:
        key = (str(point["trajectory_id"]), int(point["point_index"]))
        point_candidates = by_key[key]
        frame = pd.DataFrame(point_candidates).reindex(columns=feature_columns)
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(frame)
            classes = list(getattr(model, "classes_", [0, 1]))
            positive_index = classes.index(1) if 1 in classes else len(classes) - 1
            probabilities = [float(p[positive_index]) for p in probs]
        else:
            raw = model.decision_function(frame)
            probabilities = [float(1 / (1 + __import__("math").exp(-max(-60, min(60, x))))) for x in raw]
        # Normalize within a point so the emissions form a categorical distribution.
        total = sum(probabilities) or 1.0
        probabilities = [p / total for p in probabilities]
        probabilities_by_point.append(probabilities)
        candidate_group.append(point_candidates)
        best = max(range(len(point_candidates)), key=lambda i: probabilities[i])
        rows.append({"trajectory_id": point["trajectory_id"], "point_index": point["point_index"],
                     "classifier_edge_id": point_candidates[best]["edge_id"]})
    if len({str(p["trajectory_id"]) for p in ordered_points}) > 1:
        raise ValueError("Submit one trajectory per request so HMM transitions do not cross trips.")
    emissions = [{str(candidate["edge_id"]): prob for candidate, prob in zip(group, probs)}
                 for group, probs in zip(candidate_group, probabilities_by_point)]
    decoded = viterbi(emissions, adjacency_log_probabilities(edges), default_transition_log_prob=-12.0)
    for row, edge_id in zip(rows, decoded):
        row["decoded_edge_id"] = edge_id
    return {"point_count": len(rows), "route": rows}
