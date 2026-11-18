"""Evaluate the selected model on held-out trips with metric GPS noise and thinning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone

from .dataset import TripData, list_trips, read_trip
from .decoding import decode_trip_hmm
from .feature_pipeline import build_trip_features
from .feature_store import load_trip_feature_set
from .geo import RoadIndex
from .hmm_training import _metric_for_edges, _metric_for_classifier
from .route_labels import project_track
from .training import _fit_candidate_model, _training_matrix, model_zoo


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def _predict(model, model_name: str, data, sample_prior: float,
             population_prior: float, chunk_rows: int = 100_000) -> np.ndarray:
    from .training import _probability_for_model
    result = np.empty(data.candidate_count, dtype=np.float32)
    for start in range(0, data.candidate_count, chunk_rows):
        stop = min(data.candidate_count, start + chunk_rows)
        result[start:stop] = _probability_for_model(
            model, model_name, data.features[start:stop], sample_prior, population_prior)
    return result


def _calibrate(probability: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    score = parameters["slope"] * np.log(clipped / (1 - clipped)) + parameters["intercept"]
    output = np.empty_like(score)
    positive = score >= 0
    output[positive] = 1 / (1 + np.exp(-score[positive]))
    exp_score = np.exp(score[~positive])
    output[~positive] = exp_score / (1 + exp_score)
    return np.clip(output, 1e-7, 1 - 1e-7).astype(np.float32)


def _apply_clean_labels(variant, clean, original_indices: np.ndarray) -> None:
    targets = clean.point_true_edges[original_indices].astype(np.int64, copy=True)
    variant.point_true_edges = targets
    variant.point_route_positions = clean.point_route_positions[original_indices].copy()
    variant.point_route_distances_m = clean.point_route_distances_m[original_indices].copy()
    if variant.candidate_count:
        candidate_targets = targets[variant.candidate_point_indices]
        variant.labels = (variant.candidate_edge_ids == candidate_targets).astype(np.int8)
    else:
        variant.labels = np.empty(0, dtype=np.int8)
    misses = np.zeros(len(targets), dtype=bool)
    for point_index, (start, stop) in enumerate(zip(variant.point_offsets[:-1], variant.point_offsets[1:])):
        misses[point_index] = stop <= start or not np.any(
            variant.candidate_edge_ids[start:stop] == targets[point_index])
    variant.point_candidate_misses = misses


def _score_variant(model, model_name: str, variant, calibrator: dict[str, float],
                   decoder: dict[str, float], sample_prior: float,
                   population_prior: float) -> dict[str, Any]:
    raw = _predict(model, model_name, variant, sample_prior, population_prior)
    calibrated = _calibrate(raw, calibrator)
    classifier = _metric_for_classifier(variant, calibrated)
    decoded = decode_trip_hmm(variant, calibrated, **decoder)
    hmm = _metric_for_edges(variant, decoded)
    return {
        "classifier_top1": classifier["observation_top1_accuracy"],
        "hmm_top1": hmm["observation_top1_accuracy"],
        "classifier_route_continuity": classifier["route_continuity"],
        "hmm_route_continuity": hmm["route_continuity"],
        "candidate_miss_rate": classifier["candidate_miss_rate"],
        "observations_scored": classifier["observations_scored"],
        "candidate_misses": classifier["candidate_misses"],
        "classifier_correct": classifier["observation_correct"],
        "hmm_correct": hmm["observation_correct"],
        "continuity_pairs": classifier["route_continuity_pairs_scored"],
        "classifier_connected": classifier["connected_pairs"],
        "hmm_connected": hmm["connected_pairs"],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["observations_scored"] for row in rows)
    misses = sum(row["candidate_misses"] for row in rows)
    correct = sum(row["classifier_correct"] for row in rows)
    hmm_correct = sum(row["hmm_correct"] for row in rows)
    continuity_pairs = sum(row["continuity_pairs"] for row in rows)
    classifier_connected = sum(row["classifier_connected"] for row in rows)
    hmm_connected = sum(row["hmm_connected"] for row in rows)
    per_trip: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        per_trip.setdefault(row["trip_id"], []).append(row)
    mean_trip_classifier = float(np.mean([
        np.mean([trip_row["classifier_top1"] for trip_row in trip_rows])
        for trip_rows in per_trip.values()
    ])) if per_trip else 0.0
    mean_trip_hmm = float(np.mean([
        np.mean([trip_row["hmm_top1"] for trip_row in trip_rows])
        for trip_rows in per_trip.values()
    ])) if per_trip else 0.0
    return {
        "pooled_classifier_top1": correct / max(1, total),
        "pooled_hmm_top1": hmm_correct / max(1, total),
        "mean_per_trip_classifier_top1": mean_trip_classifier,
        "mean_per_trip_hmm_top1": mean_trip_hmm,
        "candidate_miss_rate": misses / max(1, total),
        "classifier_route_continuity": classifier_connected / max(1, continuity_pairs),
        "hmm_route_continuity": hmm_connected / max(1, continuity_pairs),
        "observations_scored": total,
        "candidate_misses": misses,
        "trip_count": len(per_trip),
    }


def _add_counts(row: dict[str, Any], classifier_metric: dict[str, Any], hmm_metric: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "classifier_correct": classifier_metric["observation_correct"],
        "hmm_correct": hmm_metric["observation_correct"],
        "continuity_pairs": classifier_metric["route_continuity_pairs_scored"],
        "classifier_connected": classifier_metric["connected_pairs"],
        "hmm_connected": hmm_metric["connected_pairs"],
    }


def _subsample_indices(point_count: int, step: int) -> np.ndarray:
    indices = np.arange(0, point_count, step, dtype=np.int64)
    if len(indices) and indices[-1] != point_count - 1:
        indices = np.append(indices, point_count - 1)
    return indices


def run_sensitivity(cache_dir: str | Path, archive_path: str | Path,
                    cv_dir: str | Path, output_dir: str | Path,
                    model_name: str, *, seed: int = 42,
                    noise_repeats: int = 3) -> dict[str, Any]:
    """Run noise/thinning checks on outer-fold held-out training trips only."""
    cache_dir, archive_path = Path(cache_dir), Path(archive_path)
    cv_dir, output_dir = Path(cv_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sensitivity_summary.json"
    csv_path = output_dir / "sensitivity_per_trip.csv"
    if summary_path.exists() or csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite sensitivity output at {output_dir}")
    plan = json.loads((cv_dir / "fold_plan.json").read_text(encoding="utf-8"))
    train_paths = sorted((cache_dir / "train").glob("*.npz"))
    fold_by_trip = {}
    for fold in plan["folds"]:
        for trip_id in fold["validation_trip_ids"]:
            fold_by_trip[trip_id] = int(fold["fold"])
    refs = {ref.trip_id: ref for ref in list_trips(archive_path, split="train")}
    fields = ["fold", "trip_id", "setting", "noise_m", "repeat", "point_stride",
              "classifier_top1", "hmm_top1", "candidate_miss_rate",
              "classifier_route_continuity", "hmm_route_continuity", "observations_scored",
              "candidate_misses", "seed"]
    all_rows: list[dict[str, Any]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=fields)
        writer.writeheader()
        csv_stream.flush()

        for fold_info in plan["folds"]:
            fold_number = int(fold_info["fold"])
            fold_record = json.loads((cv_dir / "folds" / model_name / f"fold_{fold_number}.json").read_text(encoding="utf-8"))
            calibrator = fold_record["classifier_platt_calibrated"]["calibrator"]
            decoder = fold_record["classifier_plus_hmm"]["decoder_params"]
            fit_seed = int(fold_record["seed"])
            validation_ids = fold_info["validation_trip_ids"]
            outer_train_paths = [path for path in train_paths if path.stem not in set(validation_ids)]
            matrix = _training_matrix(outer_train_paths, seed + fold_number * 10000)
            estimator = clone(model_zoo(fit_seed)[model_name])
            model, sample_prior, population_prior = _fit_candidate_model(
                estimator, model_name, *matrix[:3], matrix[3], fit_seed)

            fold_rows: list[dict[str, Any]] = []
            for trip_id in validation_ids:
                ref = refs[trip_id]
                raw_trip = read_trip(archive_path, ref)
                clean = load_trip_feature_set(cache_dir / "train" / f"{trip_id}.npz")
                track_xy, zone, epsg = project_track(raw_trip)
                road = RoadIndex.build(raw_trip.nodes, raw_trip.arcs, zone)

                with np.load(cv_dir / "predictions" / model_name / f"fold_{fold_number}" / f"{trip_id}.npz", allow_pickle=False) as stored:
                    raw_prob = stored["calibrated_probability"].copy()
                    raw_edges = stored["calibrated_edges"].copy()
                    hmm_edges = stored["hmm_edges"].copy()
                clean_classifier = _metric_for_classifier(clean, raw_prob)
                clean_hmm = _metric_for_edges(clean, hmm_edges)
                clean_row = {
                    "fold": fold_number, "trip_id": trip_id, "setting": "clean_oof",
                    "noise_m": 0.0, "repeat": 0, "point_stride": 1,
                    "classifier_top1": clean_classifier["observation_top1_accuracy"],
                    "hmm_top1": clean_hmm["observation_top1_accuracy"],
                    "candidate_miss_rate": clean_classifier["candidate_miss_rate"],
                    "classifier_route_continuity": clean_classifier["route_continuity"],
                    "hmm_route_continuity": clean_hmm["route_continuity"],
                    "observations_scored": clean.point_count,
                    "candidate_misses": clean_classifier["candidate_misses"],
                    "seed": seed,
                }
                fold_rows.append(_add_counts(clean_row, clean_classifier, clean_hmm))

                for noise_m in (5.0, 10.0, 20.0):
                    for repeat in range(noise_repeats):
                        variant_seed = seed + sum(map(ord, trip_id)) * 100 + int(noise_m) * 10 + repeat
                        rng = np.random.default_rng(variant_seed)
                        noisy_xy = track_xy + rng.normal(0.0, noise_m, size=track_xy.shape)
                        variant = build_trip_features(
                            raw_trip, radius_m=250.0, max_candidates=128,
                            include_labels=False, track_xy_override=noisy_xy,
                            road_override=road,
                        )
                        _apply_clean_labels(variant, clean, np.arange(clean.point_count))
                        score = _score_variant(model, model_name, variant, calibrator, decoder,
                                               sample_prior, population_prior)
                        row = {"fold": fold_number, "trip_id": trip_id,
                               "setting": "gaussian_noise", "noise_m": noise_m,
                               "repeat": repeat, "point_stride": 1, "seed": variant_seed, **score}
                        fold_rows.append(row)
                        del variant

                for stride in (2, 5, 10):
                    indices = _subsample_indices(clean.point_count, stride)
                    thinned_trip = TripData(raw_trip.split, raw_trip.trip_id,
                        raw_trip.track[indices].copy(), raw_trip.nodes, raw_trip.arcs, raw_trip.route)
                    variant = build_trip_features(
                        thinned_trip, radius_m=250.0, max_candidates=128,
                        include_labels=False, track_xy_override=track_xy[indices],
                        road_override=road,
                    )
                    _apply_clean_labels(variant, clean, indices)
                    score = _score_variant(model, model_name, variant, calibrator, decoder,
                                           sample_prior, population_prior)
                    row = {"fold": fold_number, "trip_id": trip_id,
                           "setting": "point_subsampling", "noise_m": 0.0,
                           "repeat": 0, "point_stride": stride, "seed": seed, **score}
                    fold_rows.append(row)
                    del variant, thinned_trip

                all_rows.extend(fold_rows)
                writer.writerows([{field: row.get(field) for field in fields} for row in fold_rows])
                csv_stream.flush()
                del clean, raw_trip, road, track_xy, raw_prob, raw_edges, hmm_edges

            fold_summary = _summarize_sensitivity_rows(fold_rows)
            _atomic_json(output_dir / f"fold_{fold_number}.json", {
                "fold": fold_number, "model": model_name,
                "calibration_parameters": calibrator,
                "decoder_parameters": decoder,
                "seed": seed, "noise_repeats": noise_repeats,
                "metrics": fold_summary,
            })
            _atomic_json(summary_path, {
                "model": model_name, "protocol": "held-out training trips from the nested GroupKFold run; student-test never used",
                "seed": seed, "noise_repeats": noise_repeats,
                "completed_folds": fold_number,
                "metrics": _summarize_sensitivity_rows(all_rows),
                "folds": [str(output_dir / f"fold_{i}.json") for i in range(1, fold_number + 1)],
            })
            print(f"[sensitivity fold {fold_number}/5] saved {len(fold_rows)} held-out trip variants", flush=True)
            del model, matrix

    return json.loads(summary_path.read_text(encoding="utf-8"))


def _summarize_sensitivity_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["setting"], float(row["noise_m"]), int(row["point_stride"]))
        groups.setdefault(key, []).append(row)
    result = {}
    for (setting, noise, stride), group_rows in sorted(groups.items()):
        total = sum(row["observations_scored"] for row in group_rows)
        pooled_classifier = sum(row["classifier_correct"] for row in group_rows) / max(1, total)
        pooled_hmm = sum(row["hmm_correct"] for row in group_rows) / max(1, total)
        trips: dict[str, list[dict[str, Any]]] = {}
        for row in group_rows:
            trips.setdefault(row["trip_id"], []).append(row)
        mean_trip_classifier = float(np.mean([
            np.mean([item["classifier_top1"] for item in trip_rows]) for trip_rows in trips.values()
        ])) if trips else 0.0
        mean_trip_hmm = float(np.mean([
            np.mean([item["hmm_top1"] for item in trip_rows]) for trip_rows in trips.values()
        ])) if trips else 0.0
        result[f"{setting}_noise{noise:g}_stride{stride}"] = {
            "pooled_classifier_top1": pooled_classifier,
            "pooled_hmm_top1": pooled_hmm,
            "mean_per_trip_classifier_top1": mean_trip_classifier,
            "mean_per_trip_hmm_top1": mean_trip_hmm,
            "candidate_miss_rate": sum(row["candidate_misses"] for row in group_rows) / max(1, total),
            "observations_scored": total,
            "trip_count": len(trips),
            "variant_count": len(group_rows),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/interim/features")
    parser.add_argument("--archive", default="data/raw/Student folder.zip")
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-repeats", type=int, default=3)
    args = parser.parse_args()
    result = run_sensitivity(args.cache_dir, args.archive, args.cv_dir,
                             args.output_dir, args.model, seed=args.seed,
                             noise_repeats=args.noise_repeats)
    print(json.dumps({"model": result["model"], "completed_folds": result["completed_folds"],
                      "setting_count": len(result["metrics"])}, indent=2))


if __name__ == "__main__":
    main()
