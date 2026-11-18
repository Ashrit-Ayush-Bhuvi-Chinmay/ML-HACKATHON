"""Training-only radius/candidate-cap sweep with incremental experiment logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .dataset import list_trips, read_trip
from .feature_pipeline import candidate_sweep


def run_training_sweep(archive_path: str | Path, output_dir: str | Path,
                       miss_target: float = 0.005,
                       radii_m: tuple[int, ...] = (50, 75, 100, 150, 200, 250),
                       caps: tuple[int | None, ...] = (8, 16, 32, None),
                       max_radius_m: float = 250.0) -> dict[str, Any]:
    """Choose candidate settings using only the 80 labelled training trips."""
    refs = list_trips(archive_path, split="train")
    if len(refs) != 80:
        raise ValueError(f"Expected 80 train trips from the supplied dataset guide; found {len(refs)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "candidate_sweep_train.csv"
    fields = ["trip_id", "radius_m", "max_candidates", "points", "candidate_miss_rate",
              "mean_candidates_before_cap", "mean_candidates_after_cap", "p95_candidates_before_cap",
              "route_label_distance_p50_m", "route_label_distance_p95_m", "route_label_distance_max_m",
              "route_positions_covered"]
    aggregate: dict[tuple[int, int | None], dict[str, float]] = {}
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for ref in refs:
            trip = read_trip(archive_path, ref)
            rows = candidate_sweep(trip, max_radius_m=max_radius_m,
                                   radii_m=tuple(float(r) for r in radii_m), caps=caps)
            for row in rows:
                writer.writerow({**row, "max_candidates": "unlimited" if row["max_candidates"] is None else row["max_candidates"]})
                key = (int(row["radius_m"]), row["max_candidates"])
                acc = aggregate.setdefault(key, {"points": 0.0, "misses": 0.0,
                    "candidate_count_before": 0.0, "candidate_count_after": 0.0})
                points = int(row["points"])
                acc["points"] += points
                acc["misses"] += float(row["candidate_miss_rate"]) * points
                acc["candidate_count_before"] += float(row["mean_candidates_before_cap"]) * points
                acc["candidate_count_after"] += float(row["mean_candidates_after_cap"]) * points
            stream.flush()
    summaries = []
    for (radius, cap), values in sorted(aggregate.items(), key=lambda kv: (kv[0][0], 10**9 if kv[0][1] is None else kv[0][1])):
        n = max(1.0, values["points"])
        summaries.append({"radius_m": radius, "max_candidates": cap,
                          "candidate_miss_rate": values["misses"] / n,
                          "mean_candidates_before_cap": values["candidate_count_before"] / n,
                          "mean_candidates_after_cap": values["candidate_count_after"] / n,
                          "training_observations": int(values["points"])})
    eligible = [s for s in summaries if s["candidate_miss_rate"] <= miss_target]
    chosen = min(eligible, key=lambda s: (s["mean_candidates_after_cap"], s["radius_m"],
                  10**9 if s["max_candidates"] is None else s["max_candidates"])) if eligible else min(
                      summaries, key=lambda s: (s["candidate_miss_rate"], s["mean_candidates_after_cap"]))
    result = {"selection_partition": "train_only", "train_trip_count": len(refs),
              "miss_target": miss_target, "options": summaries, "selected": chosen,
              "route_labeling_method": "monotone nearest route arc; use a short forward window normally, expand to remaining suffix when local match distance exceeds 50 m, with 15 m cost per skipped route arc",
              "selection_rule": "lowest mean retained candidates among options meeting candidate miss target; otherwise minimum miss rate"}
    (output_dir / "candidate_selection_train.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def select_from_sweep_log(detail_csv: str | Path, output_json: str | Path,
                          miss_target: float = 0.01) -> dict[str, Any]:
    """Re-select settings from a completed training-only sweep without rerunning geometry."""
    aggregate: dict[tuple[int, int | None], dict[str, float]] = {}
    with Path(detail_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            radius = int(float(row["radius_m"]))
            cap = None if row["max_candidates"] == "unlimited" else int(row["max_candidates"])
            key = radius, cap
            acc = aggregate.setdefault(key, {"points": 0.0, "misses": 0.0,
                "candidate_count_before": 0.0, "candidate_count_after": 0.0})
            points = int(row["points"])
            acc["points"] += points
            acc["misses"] += float(row["candidate_miss_rate"]) * points
            acc["candidate_count_before"] += float(row["mean_candidates_before_cap"]) * points
            acc["candidate_count_after"] += float(row["mean_candidates_after_cap"]) * points
    summaries = []
    for (radius, cap), values in aggregate.items():
        n = max(1.0, values["points"])
        summaries.append({"radius_m": radius, "max_candidates": cap,
                          "candidate_miss_rate": values["misses"] / n,
                          "mean_candidates_before_cap": values["candidate_count_before"] / n,
                          "mean_candidates_after_cap": values["candidate_count_after"] / n,
                          "training_observations": int(values["points"])})
    eligible = [s for s in summaries if s["candidate_miss_rate"] <= miss_target]
    if not eligible:
        raise ValueError(f"No candidate option meets the requested {miss_target:.2%} training miss target.")
    chosen = min(eligible, key=lambda s: (s["mean_candidates_after_cap"], s["radius_m"],
                   10**9 if s["max_candidates"] is None else s["max_candidates"]))
    result = {"selection_partition": "train_only", "miss_target": miss_target,
              "options": sorted(summaries, key=lambda s: (s["radius_m"],
                  10**9 if s["max_candidates"] is None else s["max_candidates"])),
              "selected": chosen,
              "route_labeling_method": "monotone nearest route arc; use a short forward window normally, expand to remaining suffix when local match distance exceeds 50 m, with 15 m cost per skipped route arc",
              "selection_rule": "lowest mean retained candidates among options meeting the train-only miss target"}
    Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
