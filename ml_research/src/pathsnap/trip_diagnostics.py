"""Nearest-arc direction and route-label alignment diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import list_trips, read_trip
from .feature_pipeline import FEATURE_COLUMNS
from .feature_store import load_trip_feature_set
from .training import _save_json
from .visualization import save_trip_diagnostic_map

FOCUS_TRIPS = ("00000054", "00000033", "00000021", "00000042")


def _trip_nearest_direction_stats(data) -> dict:
    result = {"trip_id": data.trip_id, "observations": data.point_count,
        "candidate_misses": 0, "baseline_errors": 0,
        "baseline_errors_with_true_arc_available": 0, "reverse_direction_errors": 0}
    for point_idx, true_edge_value in enumerate(data.point_true_edges):
        true_edge = int(true_edge_value)
        start, stop = int(data.point_offsets[point_idx]), int(data.point_offsets[point_idx + 1])
        if start == stop:
            continue
        candidate_ids = data.candidate_edge_ids[start:stop]
        true_present = bool(np.any(candidate_ids == true_edge))
        result["candidate_misses"] += int(not true_present)
        nearest_row = start + int(np.argmin(data.candidate_distances_m[start:stop]))
        predicted_edge = int(data.candidate_edge_ids[nearest_row])
        if predicted_edge == true_edge:
            continue
        result["baseline_errors"] += 1
        if not true_present:
            continue
        result["baseline_errors_with_true_arc_available"] += 1
        true_from, true_to = map(int, data.arcs[true_edge])
        pred_from, pred_to = map(int, data.arcs[predicted_edge])
        result["reverse_direction_errors"] += int(pred_from == true_to and pred_to == true_from)
    result["reverse_share_among_checkable_errors"] = (
        result["reverse_direction_errors"] /
        max(1, result["baseline_errors_with_true_arc_available"]))
    result["reverse_share_among_all_baseline_errors"] = (
        result["reverse_direction_errors"] / max(1, result["baseline_errors"]))
    result["nearest_top1"] = 1.0 - result["baseline_errors"] / max(1, result["observations"])
    result["candidate_miss_rate"] = result["candidate_misses"] / max(1, result["observations"])
    return result


def _alignment_stats(data) -> dict:
    true_candidate_distances, true_candidate_ranks, heading_deltas = [], [], []
    nearest_distances = []
    heading_column = FEATURE_COLUMNS.index("heading_delta_deg")
    for point_idx, true_edge_value in enumerate(data.point_true_edges):
        start, stop = int(data.point_offsets[point_idx]), int(data.point_offsets[point_idx + 1])
        if start == stop:
            continue
        nearest_distances.append(float(np.min(data.candidate_distances_m[start:stop])))
        local_ids = data.candidate_edge_ids[start:stop]
        found = np.flatnonzero(local_ids == int(true_edge_value))
        if len(found):
            row = start + int(found[0])
            true_candidate_ranks.append(int(found[0]) + 1)
            true_candidate_distances.append(float(data.candidate_distances_m[row]))
            heading_deltas.append(abs(float(data.features[row, heading_column])))
    def percentiles(values):
        return [float(x) for x in np.percentile(values, [50, 95])] if values else [None, None]
    label_p50, label_p95 = percentiles(data.point_route_distances_m.tolist())
    nearest_p50, nearest_p95 = percentiles(nearest_distances)
    true_candidate_p50, true_candidate_p95 = percentiles(true_candidate_distances)
    heading_p50, heading_p95 = percentiles(heading_deltas)
    return {"trip_id": data.trip_id, "points": data.point_count,
        "candidate_miss_rate": float(np.mean(data.point_candidate_misses)),
        "nearest_baseline_top1": float(np.mean([
            int(data.candidate_edge_ids[int(data.point_offsets[i]) +
                int(np.argmin(data.candidate_distances_m[int(data.point_offsets[i]):int(data.point_offsets[i+1])]))])
            == int(data.point_true_edges[i])
            for i in range(data.point_count) if data.point_offsets[i+1] > data.point_offsets[i]])),
        "assigned_label_distance_p50_p95_m": [label_p50, label_p95],
        "nearest_candidate_distance_p50_p95_m": [nearest_p50, nearest_p95],
        "true_candidate_distance_p50_p95_m": [true_candidate_p50, true_candidate_p95],
        "true_candidate_rank_median": float(np.median(true_candidate_ranks)) if true_candidate_ranks else None,
        "absolute_heading_difference_p50_p95_deg": [heading_p50, heading_p95],
        "assigned_route_positions_unique": int(len(np.unique(data.point_route_positions))),
        "label_assumption": "per-point labels are a monotone nearest-geometry assignment from the route arc sequence"}


def diagnose_trips(archive_path: str | Path, cache_dir: str | Path,
                  output_dir: str | Path, trip_ids=FOCUS_TRIPS,
                  create_maps: bool = True) -> dict:
    """Report reverse-edge shares and map two difficult trips without changing labels."""
    archive_path, cache_dir, output_dir = Path(archive_path), Path(cache_dir), Path(output_dir)
    selected_ids = tuple(str(t) for t in trip_ids)
    refs = {ref.trip_id: ref for ref in list_trips(archive_path, split="train")}
    details = {}
    for trip_id in selected_ids:
        if trip_id not in refs:
            raise ValueError(f"Training trip {trip_id} was not found in the supplied archive.")
        data = load_trip_feature_set(cache_dir / "train" / f"{trip_id}.npz")
        details[trip_id] = {"reverse_direction": _trip_nearest_direction_stats(data),
                            "alignment": _alignment_stats(data)}
    all_path_stats = []
    for path in sorted((cache_dir / "train").glob("*.npz")):
        data = load_trip_feature_set(path)
        all_path_stats.append(_trip_nearest_direction_stats(data))
    overall = {key: sum(row[key] for row in all_path_stats) for key in (
        "observations", "candidate_misses", "baseline_errors",
        "baseline_errors_with_true_arc_available", "reverse_direction_errors")}
    overall["reverse_share_among_checkable_errors"] = (
        overall["reverse_direction_errors"] / max(1, overall["baseline_errors_with_true_arc_available"]))
    overall["reverse_share_among_all_baseline_errors"] = (
        overall["reverse_direction_errors"] / max(1, overall["baseline_errors"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"overall_training": overall, "focus_trip_details": details,
              "label_changes_made": False, "maps": []}
    _save_json(output_dir / "trip_label_diagnostics.json", result)
    if create_maps:
        for trip_id in ("00000054", "00000033"):
            trip = read_trip(archive_path, refs[trip_id])
            features = load_trip_feature_set(cache_dir / "train" / f"{trip_id}.npz")
            map_path = output_dir / f"trip_{trip_id}_label_vs_nearest.html"
            save_trip_diagnostic_map(trip, features, map_path)
            result["maps"].append(str(map_path))
        _save_json(output_dir / "trip_label_diagnostics.json", result)
    return result
