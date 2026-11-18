"""Metric-space candidate generation and compact per-(point, arc) feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import TripData
from .geo import RoadIndex, heading_delta, point_heading_and_speed
from .route_labels import assign_route_labels, project_track

FEATURE_COLUMNS = (
    "distance_m", "candidate_rank", "along_fraction", "heading_delta_deg", "speed_mps",
    "arc_length_m", "from_degree", "to_degree", "node_degree_sum", "candidate_count",
    "nearest_candidate_gap_m", "previous_projection_delta_m", "connects_previous_candidate",
    "elapsed_s", "gap_reset",
)


@dataclass
class TripFeatureSet:
    split: str
    trip_id: str
    features: np.ndarray
    labels: np.ndarray
    candidate_edge_ids: np.ndarray
    candidate_from_nodes: np.ndarray
    candidate_to_nodes: np.ndarray
    candidate_point_indices: np.ndarray
    candidate_distances_m: np.ndarray
    candidate_along_fractions: np.ndarray
    candidate_arc_lengths_m: np.ndarray
    point_offsets: np.ndarray
    point_timestamps_s: np.ndarray
    point_true_edges: np.ndarray
    point_route_positions: np.ndarray
    point_route_distances_m: np.ndarray
    point_observed_steps_m: np.ndarray
    point_gap_resets: np.ndarray
    point_candidate_misses: np.ndarray
    arcs: np.ndarray
    epsg: str

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_edge_ids)

    @property
    def point_count(self) -> int:
        return len(self.point_true_edges)


def _projection_fraction(point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    vector = end_xy - start_xy
    denom = float(vector @ vector)
    return 0.0 if denom == 0 else float(np.clip(((point_xy - start_xy) @ vector) / denom, 0.0, 1.0))


def _connects(previous_id: int, current_id: int, arcs: np.ndarray) -> bool:
    return previous_id == current_id or int(arcs[previous_id, 1]) == int(arcs[current_id, 0])


def build_trip_features(trip: TripData, radius_m: float = 150.0,
                        max_candidates: int | None = 32,
                        sample_spacing_m: float = 40.0,
                        gap_reset_seconds: float = 30.0,
                        include_labels: bool = True,
                        track_xy_override: np.ndarray | None = None,
                        road_override: RoadIndex | None = None) -> TripFeatureSet:
    """Build exact-distance candidates via a KD-tree over densified projected road arcs.

    Route labels are separate target/metadata arrays and are never included in ``features``.
    """
    projected_track_xy, zone, epsg = project_track(trip)
    if track_xy_override is None:
        track_xy = projected_track_xy
    else:
        track_xy = np.asarray(track_xy_override, dtype=np.float64)
        if track_xy.shape != projected_track_xy.shape or not np.isfinite(track_xy).all():
            raise ValueError("track_xy_override must be finite metric coordinates with one row per GPS point")
    road = road_override or RoadIndex.build(trip.nodes, trip.arcs, zone, sample_spacing_m)
    if include_labels:
        true_edges, route_positions, route_distances = assign_route_labels(
            trip, road, track_xy, gap_reset_seconds)
    else:
        true_edges = np.full(len(track_xy), -1, dtype=np.int64)
        route_positions = np.full(len(track_xy), -1, dtype=np.int64)
        route_distances = np.full(len(track_xy), np.nan, dtype=np.float64)

    capacity = None if max_candidates is None else len(track_xy) * max_candidates
    if capacity is None:
        x_rows: Any = []
        labels: Any = []
        edge_ids: Any = []
        from_nodes: Any = []
        to_nodes: Any = []
        candidate_points: Any = []
        candidate_distances: Any = []
        candidate_fractions: Any = []
        candidate_lengths: Any = []
    else:
        x_rows = np.empty((capacity, len(FEATURE_COLUMNS)), dtype=np.float32)
        labels = np.empty(capacity, dtype=np.int8)
        edge_ids = np.empty(capacity, dtype=np.int64)
        from_nodes = np.empty(capacity, dtype=np.int64)
        to_nodes = np.empty(capacity, dtype=np.int64)
        candidate_points = np.empty(capacity, dtype=np.int32)
        candidate_distances = np.empty(capacity, dtype=np.float32)
        candidate_fractions = np.empty(capacity, dtype=np.float32)
        candidate_lengths = np.empty(capacity, dtype=np.float32)
    cursor = 0
    point_offsets = [0]
    point_observed_steps: list[float] = []
    point_gap_resets: list[bool] = []
    point_candidate_misses: list[bool] = []
    prior_candidates: list[dict[str, Any]] = []
    prior_edge_ids: set[int] = set()
    prior_to_nodes: set[int] = set()
    best_previous_count = 8

    for i, point_xy in enumerate(track_xy):
        elapsed = float(trip.track[i, 2] - trip.track[i-1, 2]) if i else 0.0
        gap_reset = bool(i and elapsed > gap_reset_seconds)
        previous_xy = track_xy[i-1] if i and not gap_reset else None
        heading, speed, observed_step = (point_heading_and_speed(previous_xy, point_xy, elapsed)
                                         if previous_xy is not None else (None, None, 0.0))
        if gap_reset:
            prior_candidates = []
            prior_edge_ids = set()
            prior_to_nodes = set()
        candidates_all = road.query(point_xy, radius_m, max_candidates=None)
        candidates = candidates_all if max_candidates is None else candidates_all[:max_candidates]
        nearest_distance = float(candidates[0]["distance_m"]) if candidates else 0.0
        true_id = int(true_edges[i])
        point_candidate_misses.append(bool(include_labels and all(c["edge_id"] != true_id for c in candidates)))
        for candidate in candidates:
            edge_id = candidate["edge_id"]
            if previous_xy is not None:
                previous_fraction = _projection_fraction(previous_xy, road.start_xy[edge_id], road.end_xy[edge_id])
                previous_delta = abs(candidate["along_fraction"] - previous_fraction) * candidate["arc_length_m"]
            else:
                previous_delta = 0.0
            connects = edge_id in prior_edge_ids or int(trip.arcs[edge_id, 0]) in prior_to_nodes
            from_degree, to_degree = candidate["from_degree"], candidate["to_degree"]
            feature_row = [
                candidate["distance_m"], candidate["rank"], candidate["along_fraction"],
                heading_delta(heading, candidate["bearing_deg"]), 0.0 if speed is None else speed,
                candidate["arc_length_m"], from_degree, to_degree, from_degree + to_degree,
                len(candidates_all), candidate["distance_m"] - nearest_distance,
                previous_delta, float(connects), min(max(elapsed, 0.0), 3600.0),
                float(gap_reset),
            ]
            values = (feature_row, int(edge_id == true_id) if include_labels else -1,
                      edge_id, int(trip.arcs[edge_id, 0]), int(trip.arcs[edge_id, 1]), i,
                      float(candidate["distance_m"]), float(candidate["along_fraction"]),
                      float(candidate["arc_length_m"]))
            if capacity is None:
                x_rows.append(values[0])
                labels.append(values[1])
                edge_ids.append(values[2])
                from_nodes.append(values[3])
                to_nodes.append(values[4])
                candidate_points.append(values[5])
                candidate_distances.append(values[6])
                candidate_fractions.append(values[7])
                candidate_lengths.append(values[8])
            else:
                x_rows[cursor], labels[cursor], edge_ids[cursor] = values[0], values[1], values[2]
                from_nodes[cursor], to_nodes[cursor] = values[3], values[4]
                candidate_points[cursor], candidate_distances[cursor] = values[5], values[6]
                candidate_fractions[cursor], candidate_lengths[cursor] = values[7], values[8]
            cursor += 1
        point_offsets.append(cursor)
        point_observed_steps.append(observed_step)
        point_gap_resets.append(gap_reset)
        prior_candidates = candidates
        # Use the eight closest prior arcs as a compact proxy for the prior candidate
        # distribution's most plausible states, rather than connecting to every nearby road.
        previous_best = candidates[:best_previous_count]
        prior_edge_ids = {int(candidate["edge_id"]) for candidate in previous_best}
        prior_to_nodes = {int(trip.arcs[candidate["edge_id"], 1]) for candidate in previous_best}

    if capacity is None:
        features = np.asarray(x_rows, dtype=np.float32).reshape((-1, len(FEATURE_COLUMNS)))
        labels = np.asarray(labels, dtype=np.int8)
        edge_ids = np.asarray(edge_ids, dtype=np.int64)
        from_nodes = np.asarray(from_nodes, dtype=np.int64)
        to_nodes = np.asarray(to_nodes, dtype=np.int64)
        candidate_points = np.asarray(candidate_points, dtype=np.int32)
        candidate_distances = np.asarray(candidate_distances, dtype=np.float32)
        candidate_fractions = np.asarray(candidate_fractions, dtype=np.float32)
        candidate_lengths = np.asarray(candidate_lengths, dtype=np.float32)
    else:
        features, labels, edge_ids = x_rows[:cursor].copy(), labels[:cursor].copy(), edge_ids[:cursor].copy()
        from_nodes, to_nodes = from_nodes[:cursor].copy(), to_nodes[:cursor].copy()
        candidate_points = candidate_points[:cursor].copy()
        candidate_distances, candidate_fractions = candidate_distances[:cursor].copy(), candidate_fractions[:cursor].copy()
        candidate_lengths = candidate_lengths[:cursor].copy()
    return TripFeatureSet(
        split=trip.split, trip_id=trip.trip_id, features=features,
        labels=labels, candidate_edge_ids=edge_ids,
        candidate_from_nodes=from_nodes, candidate_to_nodes=to_nodes,
        candidate_point_indices=candidate_points,
        candidate_distances_m=candidate_distances,
        candidate_along_fractions=candidate_fractions,
        candidate_arc_lengths_m=candidate_lengths,
        point_offsets=np.asarray(point_offsets, dtype=np.int64),
        point_timestamps_s=trip.track[:, 2].astype(np.float64),
        point_true_edges=true_edges, point_route_positions=route_positions,
        point_route_distances_m=route_distances,
        point_observed_steps_m=np.asarray(point_observed_steps, dtype=np.float32),
        point_gap_resets=np.asarray(point_gap_resets, dtype=bool),
        point_candidate_misses=np.asarray(point_candidate_misses, dtype=bool),
        arcs=trip.arcs, epsg=epsg,
    )


def candidate_sweep(trip: TripData, max_radius_m: float = 250.0,
                    radii_m: tuple[float, ...] = (50, 75, 100, 150, 200, 250),
                    caps: tuple[int | None, ...] = (8, 16, 32, None),
                    sample_spacing_m: float = 40.0,
                    gap_reset_seconds: float = 30.0) -> list[dict[str, Any]]:
    """Measure true-arc candidate miss rates for radius/cap options on one permitted trip."""
    track_xy, zone, _ = project_track(trip)
    road = RoadIndex.build(trip.nodes, trip.arcs, zone, sample_spacing_m)
    true_edges, route_positions, route_distances = assign_route_labels(trip, road, track_xy, gap_reset_seconds)
    point_candidate_distances = []
    true_distances_and_ranks = []
    for point_xy, true_id in zip(track_xy, true_edges):
        candidates = road.query(point_xy, max_radius_m, max_candidates=None)
        point_candidate_distances.append([row["distance_m"] for row in candidates])
        true_distances_and_ranks.append(next(((row["distance_m"], rank) for rank, row in enumerate(candidates)
                                             if row["edge_id"] == int(true_id)), (None, None)))
    total = max(1, len(true_distances_and_ranks))
    results = []
    for radius in radii_m:
        counts = [sum(d <= radius for d in distances) for distances in point_candidate_distances]
        for cap in caps:
            misses = sum(distance is None or distance > radius or (cap is not None and rank >= cap)
                         for distance, rank in true_distances_and_ranks)
            results.append({"trip_id": trip.trip_id, "radius_m": radius, "max_candidates": cap,
                            "points": len(true_distances_and_ranks), "candidate_miss_rate": misses / total,
                            "mean_candidates_before_cap": float(np.mean(counts)),
                            "mean_candidates_after_cap": float(np.mean([n if cap is None else min(n, cap) for n in counts])),
                            "p95_candidates_before_cap": float(np.percentile(counts, 95)) if counts else 0.0,
                            "route_label_distance_p50_m": float(np.percentile(route_distances, 50)),
                            "route_label_distance_p95_m": float(np.percentile(route_distances, 95)),
                            "route_label_distance_max_m": float(np.max(route_distances)),
                            "route_positions_covered": int(len(np.unique(route_positions)))})
    return results
