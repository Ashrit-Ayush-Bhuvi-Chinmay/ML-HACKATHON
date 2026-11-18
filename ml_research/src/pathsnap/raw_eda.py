"""Reproducible EDA for the supplied ZIP archive, using only training/student partitions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any
from zipfile import ZipFile

import numpy as np

from .dataset import TripData, list_trips, read_trip
from .geo import project_utm, utm_zone

LARGE_GAP_SECONDS = 30.0


def _weak_components(node_count: int, arcs: np.ndarray) -> tuple[int, int]:
    parent = list(range(node_count))
    rank = [0] * node_count

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    degree = np.zeros(node_count, dtype=np.int64)
    for u, v in arcs:
        u, v = int(u), int(v)
        union(u, v)
        degree[u] += 1
        degree[v] += 1
    roots = {find(i) for i in range(node_count)}
    return len(roots), int(np.count_nonzero(degree == 0))


def inspect_trip(trip: TripData, file_sizes: dict[str, int]) -> dict[str, Any]:
    track, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    diffs = np.diff(track[:, 2]) if len(track) > 1 else np.asarray([], dtype=float)
    median_lon = float(np.median(track[:, 0]))
    median_lat = float(np.median(track[:, 1]))
    zone = utm_zone(median_lon)
    route_breaks = sum(int(arcs[int(a), 1]) != int(arcs[int(b), 0])
                       for a, b in zip(route, route[1:]))
    component_count, isolated_count = _weak_components(len(nodes), arcs)
    projected = np.asarray([project_utm(lon, lat, zone)[:2] for lon, lat in track[:, :2]])
    max_step_m = float(np.linalg.norm(np.diff(projected, axis=0), axis=1).max()) if len(projected) > 1 else 0.0
    return {
        "split": trip.split, "trip_id": trip.trip_id,
        "file_sizes_bytes": file_sizes,
        "track_points": int(len(track)), "node_count": int(len(nodes)), "arc_count": int(len(arcs)),
        "route_arc_count": int(len(route)),
        "track_bounds_lon_lat": {"lon_min": float(track[:, 0].min()), "lon_max": float(track[:, 0].max()),
                                 "lat_min": float(track[:, 1].min()), "lat_max": float(track[:, 1].max())},
        "median_center_lon_lat": [median_lon, median_lat],
        "distance_crs": f"WGS84 / UTM zone {zone}{'N' if median_lat >= 0 else 'S'}",
        "distance_epsg": (32600 if median_lat >= 0 else 32700) + zone,
        "invalid_coordinate_count": int(np.count_nonzero((np.abs(track[:, 0]) > 180) | (np.abs(track[:, 1]) > 90))),
        "timestamp": {
            "strictly_increasing": bool(np.all(diffs > 0)),
            "one_second_step_fraction": float(np.mean(np.isclose(diffs, 1.0))) if len(diffs) else None,
            "gap_gt_1s_count": int(np.count_nonzero(diffs > 1.01)),
            "gap_gt_30s_count": int(np.count_nonzero(diffs > LARGE_GAP_SECONDS)),
            "max_gap_s": float(diffs.max()) if len(diffs) else 0.0,
        },
        "max_observed_step_m": max_step_m,
        "network": {"weak_component_count_including_isolates": component_count,
                    "isolated_node_count": isolated_count,
                    "invalid_arc_endpoint_count": int(np.count_nonzero((arcs < 0) | (arcs >= len(nodes))))},
        "route": {"invalid_arc_id_count": int(np.count_nonzero((route < 0) | (route >= len(arcs)))),
                  "disconnected_consecutive_arc_pairs": route_breaks,
                  "reference_label_assumption": "Route arc sequence is mapped monotonically to GPS points by nearest geometry."},
    }


def _split_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"trip_count": 0}
    def nums(field):
        return [r[field] for r in records]
    gaps = [r["timestamp"] for r in records]
    return {
        "trip_count": len(records),
        "track_points_total": sum(nums("track_points")),
        "nodes_total": sum(nums("node_count")),
        "arcs_total": sum(nums("arc_count")),
        "route_arcs_total": sum(nums("route_arc_count")),
        "track_points_per_trip_min_median_max": [min(nums("track_points")), median(nums("track_points")), max(nums("track_points"))],
        "timestamp_gap_gt_1s_total": sum(g["gap_gt_1s_count"] for g in gaps),
        "timestamp_gap_gt_30s_total": sum(g["gap_gt_30s_count"] for g in gaps),
        "timestamp_max_gap_s": max(g["max_gap_s"] for g in gaps),
        "one_second_step_fraction_weighted": (
            sum((r["track_points"] - 1) * (r["timestamp"]["one_second_step_fraction"] or 0.0) for r in records)
            / max(1, sum(r["track_points"] - 1 for r in records))),
        "route_discontinuities_total": sum(r["route"]["disconnected_consecutive_arc_pairs"] for r in records),
        "utm_zones": sorted({int(r["distance_epsg"]) % 100 for r in records}),
    }


def profile_archive(archive_path: str | Path) -> dict[str, Any]:
    """Build an inventory and quality report without opening evaluator data (not in this archive)."""
    refs = list_trips(archive_path)
    records = []
    errors = []
    with ZipFile(archive_path) as archive:
        for ref in refs:
            sizes = {suffix: archive.getinfo(name).file_size for suffix, name in ref.files.items()}
            try:
                trip = read_trip(archive_path, ref)
                records.append(inspect_trip(trip, sizes))
            except Exception as exc:
                errors.append({"split": ref.split, "trip_id": ref.trip_id,
                               "error": f"{type(exc).__name__}: {exc}"})
    splits = {split: _split_aggregate([r for r in records if r["split"] == split])
              for split in sorted({r["split"] for r in records})}
    return {
        "source_archive": str(archive_path),
        "file_format": "headerless whitespace-delimited text",
        "schema": {"track": ["longitude", "latitude", "timestamp_seconds"],
                   "nodes": ["longitude", "latitude"],
                   "arcs": ["from_node_id", "to_node_id; row index is arc_id"],
                   "route": ["arc_id; chronological sequence, not point-aligned"]},
        "observed_splits": splits,
        "expected_but_absent_partition": "evaluator_test (10 trips; supplied archive contains train and student_test only)",
        "trip_count": len(records), "errors": errors, "trips": records,
        "assumptions": [
            "GPS/node columns are interpreted as longitude then latitude, checked against geographic bounds.",
            "Each trip is projected to the WGS84 / UTM zone selected from its median longitude and latitude before distance calculations.",
            "Route labels are assigned to GPS points by a forward-only nearest-route-arc heuristic; supplied route references may be imperfect.",
            f"HMM chains reset across timestamp gaps greater than {LARGE_GAP_SECONDS:g} seconds and at observations with no candidates.",
        ],
    }


def write_profile(archive_path: str | Path, output_json: str | Path, output_csv: str | Path) -> dict[str, Any]:
    profile = profile_archive(archive_path)
    output_json, output_csv = Path(output_json), Path(output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "trip_id", "track_points", "node_count", "arc_count", "route_arc_count",
                  "distance_epsg", "max_gap_s", "gap_gt_1s_count", "gap_gt_30s_count",
                  "isolated_node_count", "weak_component_count_including_isolates",
                  "disconnected_consecutive_arc_pairs"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in profile["trips"]:
            writer.writerow({"split": row["split"], "trip_id": row["trip_id"],
                             "track_points": row["track_points"], "node_count": row["node_count"],
                             "arc_count": row["arc_count"], "route_arc_count": row["route_arc_count"],
                             "distance_epsg": row["distance_epsg"], "max_gap_s": row["timestamp"]["max_gap_s"],
                             "gap_gt_1s_count": row["timestamp"]["gap_gt_1s_count"],
                             "gap_gt_30s_count": row["timestamp"]["gap_gt_30s_count"],
                             "isolated_node_count": row["network"]["isolated_node_count"],
                             "weak_component_count_including_isolates": row["network"]["weak_component_count_including_isolates"],
                             "disconnected_consecutive_arc_pairs": row["route"]["disconnected_consecutive_arc_pairs"]})
    return profile
