"""Incremental feature-build and compact trip-cache storage."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import list_trips, read_trip
from .feature_pipeline import FEATURE_COLUMNS, TripFeatureSet, build_trip_features

ARRAY_FIELDS = (
    "features", "labels", "candidate_edge_ids", "candidate_from_nodes", "candidate_to_nodes",
    "candidate_point_indices", "candidate_distances_m", "candidate_along_fractions",
    "candidate_arc_lengths_m", "point_offsets", "point_timestamps_s", "point_true_edges",
    "point_route_positions", "point_route_distances_m", "point_observed_steps_m",
    "point_gap_resets", "point_candidate_misses", "arcs",
)


def save_trip_feature_set(data: TripFeatureSet, path: str | Path,
                          cache_settings: dict[str, Any] | None = None) -> Path:
    """Write one trip's arrays; no raw data or model outputs are mixed into the cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: getattr(data, name) for name in ARRAY_FIELDS})
    path.with_suffix(".json").write_text(json.dumps({"cache_version": 2, "split": data.split, "trip_id": data.trip_id,
        "epsg": data.epsg, "feature_columns": list(FEATURE_COLUMNS),
        "cache_settings": cache_settings or {}}, indent=2),
        encoding="utf-8")
    return path


def load_trip_feature_set(path: str | Path) -> TripFeatureSet:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ARRAY_FIELDS}
    return TripFeatureSet(split=metadata["split"], trip_id=metadata["trip_id"],
                          epsg=metadata["epsg"], **arrays)


def build_features_for_split(archive_path: str | Path, split: str,
                             radius_m: float, max_candidates: int | None,
                             output_dir: str | Path, gap_reset_seconds: float = 30.0) -> list[Path]:
    """Build and save one trip at a time; logs each completed trip before moving on."""
    refs = list_trips(archive_path, split=split)
    if not refs:
        raise ValueError(f"No trips found for partition {split!r}")
    output_dir = Path(output_dir)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"feature_build_{split}.csv"
    fields = ["split", "trip_id", "points", "candidate_rows", "candidate_miss_rate",
              "positive_fraction", "mean_candidates", "route_label_distance_p50_m",
              "route_label_distance_p95_m", "epsg", "build_seconds", "cache_bytes"]
    paths = []
    previous_rows = {}
    if log_path.exists():
        with log_path.open(newline="", encoding="utf-8") as old_stream:
            previous_rows = {row["trip_id"]: row for row in csv.DictReader(old_stream)}
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for ordinal, ref in enumerate(refs, start=1):
            path = split_dir / f"{ref.trip_id}.npz"
            if path.exists() and path.with_suffix(".json").exists():
                try:
                    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                    expected_settings = {"radius_m": radius_m, "max_candidates": max_candidates,
                                        "gap_reset_seconds": gap_reset_seconds}
                    cache_is_current = (metadata.get("cache_version") == 2 and
                                        metadata.get("feature_columns") == list(FEATURE_COLUMNS) and
                                        metadata.get("cache_settings") == expected_settings)
                except (OSError, json.JSONDecodeError):
                    cache_is_current = False
            else:
                cache_is_current = False
            if cache_is_current:
                row = previous_rows.get(ref.trip_id)
                if row is None:
                    cached = load_trip_feature_set(path)
                    row = {"split": split, "trip_id": ref.trip_id, "points": cached.point_count,
                           "candidate_rows": cached.candidate_count,
                           "candidate_miss_rate": float(np.mean(cached.point_candidate_misses)),
                           "positive_fraction": float(np.mean(cached.labels == 1)) if cached.candidate_count else 0.0,
                           "mean_candidates": cached.candidate_count / max(1, cached.point_count),
                           "route_label_distance_p50_m": float(np.nanpercentile(cached.point_route_distances_m, 50)),
                           "route_label_distance_p95_m": float(np.nanpercentile(cached.point_route_distances_m, 95)),
                           "epsg": cached.epsg, "build_seconds": 0.0, "cache_bytes": path.stat().st_size}
                    del cached
                writer.writerow(row)
                stream.flush()
                paths.append(path)
                print(f"[{split} {ordinal}/{len(refs)}] trip={ref.trip_id} cache already complete", flush=True)
                continue
            start = time.perf_counter()
            trip = read_trip(archive_path, ref)
            data = build_trip_features(trip, radius_m=radius_m, max_candidates=max_candidates,
                                       gap_reset_seconds=gap_reset_seconds, include_labels=True)
            path = save_trip_feature_set(data, path, {"radius_m": radius_m,
                "max_candidates": max_candidates, "gap_reset_seconds": gap_reset_seconds})
            elapsed = time.perf_counter() - start
            row = {"split": split, "trip_id": ref.trip_id, "points": data.point_count,
                   "candidate_rows": data.candidate_count,
                   "candidate_miss_rate": float(np.mean(data.point_candidate_misses)),
                   "positive_fraction": float(np.mean(data.labels == 1)) if data.candidate_count else 0.0,
                   "mean_candidates": data.candidate_count / max(1, data.point_count),
                   "route_label_distance_p50_m": float(np.nanpercentile(data.point_route_distances_m, 50)),
                   "route_label_distance_p95_m": float(np.nanpercentile(data.point_route_distances_m, 95)),
                   "epsg": data.epsg, "build_seconds": elapsed, "cache_bytes": path.stat().st_size}
            writer.writerow(row)
            stream.flush()
            paths.append(path)
            print(f"[{split} {ordinal}/{len(refs)}] trip={ref.trip_id} points={data.point_count} "
                  f"candidates={data.candidate_count} miss={row['candidate_miss_rate']:.4%} "
                  f"time={elapsed:.1f}s", flush=True)
            del data, trip
    return paths
