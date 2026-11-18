"""Freeze the selected model and predict arcs directly from raw trip files.

Training, Platt calibration, and final HMM selection use only the 80 training
trips. The student comparison first writes predictions from .track/.nodes/.arcs;
it opens .route files only in a separate scoring pass after those predictions
have been saved.
"""

from __future__ import annotations

import csv
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from .dataset import TripData, list_trips, read_trip
from .decoding import decode_trip_hmm
from .feature_pipeline import FEATURE_COLUMNS, build_trip_features
from .feature_store import load_trip_feature_set
from .geo import project_utm, utm_zone
from .hmm_training import DECODER_GRID, _metric_for_classifier, _metric_for_edges
from .robust_experiment import _apply_platt, _candidate_summary, _fit_platt
from .route_labels import project_track
from .training import (
    _fit_candidate_model,
    _training_matrix,
    model_zoo,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = REPO_ROOT / "experiments" / "final_model_15features" / "pathsnap_model.joblib"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fold_by_trip(cv_dir: Path) -> dict[str, int]:
    plan = json.loads((cv_dir / "fold_plan.json").read_text(encoding="utf-8"))
    return {
        trip_id: int(fold["fold"])
        for fold in plan["folds"]
        for trip_id in fold["validation_trip_ids"]
    }


def fit_final_model(cache_dir: str | Path, cv_dir: str | Path,
                    output_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    """Fit all training trips after the grouped comparison has selected a model."""
    random.seed(seed)
    np.random.seed(seed)
    cache_dir, cv_dir, output_dir = Path(cache_dir), Path(cv_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = cv_dir / "model_comparison_15feature_nested.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"Grouped comparison is not complete: {comparison_path}")
    artifact_path = output_dir / "pathsnap_model.joblib"
    manifest_path = output_dir / "final_model_manifest.json"
    if artifact_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite a frozen final model in {output_dir}")
    progress_path = output_dir / "final_model_progress.json"
    _atomic_json(progress_path, {"stage": "loading_training_oof_predictions", "seed": seed,
                                 "student_test_opened": False})

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    model_name = comparison["selected_model_by_hmm_mean_per_trip_top1"]
    fold_by_trip = _fold_by_trip(cv_dir)
    train_paths = sorted((cache_dir / "train").glob("*.npz"))
    if len(train_paths) != 80:
        raise ValueError(f"Expected exactly 80 training caches; found {len(train_paths)}")

    oof_probabilities: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    for path in train_paths:
        trip_id = path.stem
        prediction_path = (cv_dir / "predictions" / model_name /
                           f"fold_{fold_by_trip[trip_id]}" / f"{trip_id}.npz")
        with np.load(prediction_path, allow_pickle=False) as saved:
            oof_probabilities.append(saved["raw_probability"].copy())
        data = load_trip_feature_set(path)
        oof_labels.append(data.labels.astype(np.int8, copy=True))
        del data

    calibration_start = time.perf_counter()
    calibrator = _fit_platt(oof_probabilities, oof_labels, seed)
    calibration_seconds = time.perf_counter() - calibration_start
    _atomic_json(progress_path, {"stage": "platt_calibrator_fitted", "seed": seed,
                                 "calibration_seconds": calibration_seconds,
                                 "student_test_opened": False})

    # Select the final decoder settings on out-of-fold training predictions only.
    decoder_totals = {
        json.dumps(settings, sort_keys=True): {"settings": settings, "correct": 0, "points": 0}
        for settings in DECODER_GRID
    }
    for ordinal, (path, raw_probability) in enumerate(zip(train_paths, oof_probabilities), start=1):
        data = load_trip_feature_set(path)
        calibrated = _apply_platt(raw_probability, calibrator)
        for key, score_row in decoder_totals.items():
            predicted = decode_trip_hmm(data, calibrated, **score_row["settings"])
            metric = _metric_for_edges(data, predicted)
            score_row["correct"] += int(metric["observation_correct"])
            score_row["points"] += int(metric["observations_scored"])
        del data, calibrated
        if ordinal % 10 == 0 or ordinal == len(train_paths):
            _atomic_json(progress_path, {"stage": "training_oof_hmm_selection",
                                         "completed_training_trips": ordinal,
                                         "training_trip_count": len(train_paths),
                                         "seed": seed, "student_test_opened": False})
            print(f"[final HMM selection] scored training OOF trips {ordinal}/{len(train_paths)}", flush=True)
    decoder_results = []
    for key, score_row in decoder_totals.items():
        decoder_results.append({
            "settings": score_row["settings"],
            "oof_top1": score_row["correct"] / max(1, score_row["points"]),
            "correct_points": score_row["correct"],
            "observations": score_row["points"],
            "tie_key": key,
        })
    selected_decoder = max(decoder_results, key=lambda row: (row["oof_top1"], row["tie_key"]))

    matrix_start = time.perf_counter()
    matrix = _training_matrix(train_paths, seed)
    matrix_seconds = time.perf_counter() - matrix_start
    estimator = clone(model_zoo(seed)[model_name])
    fit_start = time.perf_counter()
    fitted, sample_prior, population_prior = _fit_candidate_model(
        estimator, model_name, *matrix[:3], matrix[3], seed)
    fit_seconds = time.perf_counter() - fit_start
    artifact = {
        "model": fitted,
        "model_name": model_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "sample_prior": float(sample_prior),
        "population_prior": float(population_prior),
        "calibrator": calibrator,
        "candidate_radius_m": 250.0,
        "max_candidates": 128,
        "sample_spacing_m": 40.0,
        "gap_reset_seconds": 30.0,
        "projection": "WGS84 UTM zone from median trip longitude; hemisphere from median latitude",
        "decoder_params": selected_decoder["settings"],
        "seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)
    manifest = {
        "model": model_name,
        "training_trips": 80,
        "student_test_trips_used_for_training_or_selection": 0,
        "seed": seed,
        "feature_columns": list(FEATURE_COLUMNS),
        "candidate_generation": {"radius_m": 250.0, "max_candidates": 128,
                                 "sample_spacing_m": 40.0},
        "projection": artifact["projection"],
        "probability_calibration": {"method": "Platt logistic calibration fitted to all 80 training-trip OOF candidate probabilities",
                                    **calibrator, "fit_seconds": calibration_seconds},
        "chosen_hmm_settings": selected_decoder["settings"],
        "training_oof_hmm_grid": decoder_results,
        "training_matrix_seconds": matrix_seconds,
        "final_fit_seconds": fit_seconds,
        "artifact": str(artifact_path),
        "student_test_scoring": "not yet run by fit_final_model; use score_final_student_test once choices are fixed",
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(progress_path, {"stage": "final_model_saved", "seed": seed,
                                 "model": model_name, "artifact": str(artifact_path),
                                 "student_test_opened": False})
    del matrix, oof_probabilities, oof_labels
    return manifest


def _read_numeric(path: str | Path, columns: int, dtype) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype, ndmin=2)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"{path} must contain exactly {columns} whitespace-separated columns")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains a non-finite value")
    return values


def _predict_trip_details(track_path: str | Path, nodes_path: str | Path,
                          arcs_path: str | Path, artifact: dict[str, Any]) -> dict[str, Any]:
    track = _read_numeric(track_path, 3, np.float64)
    nodes = _read_numeric(nodes_path, 2, np.float64)
    arcs = _read_numeric(arcs_path, 2, np.int64)
    if np.any(arcs < 0) or (len(arcs) and int(arcs.max()) >= len(nodes)):
        raise ValueError("The .arcs file refers to a missing row in the .nodes file")
    if len(track) > 1 and np.any(np.diff(track[:, 2]) < 0):
        raise ValueError("Track timestamps must be nondecreasing")
    trip_id = Path(track_path).stem
    trip = TripData("inference", trip_id, track, nodes, arcs, np.empty(0, dtype=np.int64))
    features = build_trip_features(
        trip,
        radius_m=float(artifact["candidate_radius_m"]),
        max_candidates=int(artifact["max_candidates"]),
        sample_spacing_m=float(artifact["sample_spacing_m"]),
        gap_reset_seconds=float(artifact["gap_reset_seconds"]),
        include_labels=False,
    )
    from .training import _probability_for_model
    raw_probability = np.empty(features.candidate_count, dtype=np.float32)
    for start in range(0, features.candidate_count, 100_000):
        stop = min(features.candidate_count, start + 100_000)
        raw_probability[start:stop] = _probability_for_model(
            artifact["model"], artifact["model_name"], features.features[start:stop],
            artifact["sample_prior"], artifact["population_prior"],
        )
    calibrated = _apply_platt(raw_probability, artifact["calibrator"])
    classifier = np.full(features.point_count, -1, dtype=np.int64)
    for point_index, (start, stop) in enumerate(zip(features.point_offsets[:-1], features.point_offsets[1:])):
        if stop > start:
            classifier[point_index] = features.candidate_edge_ids[
                start + int(np.argmax(calibrated[start:stop]))]
    decoded = decode_trip_hmm(features, calibrated, **artifact["decoder_params"])
    return {
        "classifier_arc_ids": classifier,
        "decoded_arc_ids": decoded,
        "candidate_edge_ids": features.candidate_edge_ids.copy(),
        "candidate_point_indices": features.candidate_point_indices.copy(),
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated,
        "epsg": features.epsg,
        "point_count": features.point_count,
    }


def predict_trip(track_path: str | Path, nodes_path: str | Path,
                 arcs_path: str | Path,
                 artifact_path: str | Path = DEFAULT_ARTIFACT) -> list[int]:
    """Return one decoded directed arc ID per raw GPS point; -1 means no candidate."""
    artifact = joblib.load(artifact_path)
    result = _predict_trip_details(track_path, nodes_path, arcs_path, artifact)
    return result["decoded_arc_ids"].astype(int).tolist()


def score_final_student_test(archive_path: str | Path, cache_dir: str | Path,
                             artifact_path: str | Path, output_dir: str | Path,
                             seed: int = 42) -> dict[str, Any]:
    """Predict all student trips from raw files, then score them once against routes."""
    archive_path, cache_dir = Path(archive_path), Path(cache_dir)
    artifact_path, output_dir = Path(artifact_path), Path(output_dir)
    predictions_dir = output_dir / "raw_predictions"
    progress_path = output_dir / "student_test_progress.json"
    final_path = output_dir / "student_test_final_metrics.json"
    if final_path.exists():
        raise FileExistsError(f"Student-test score already finalized at {final_path}")
    refs = list_trips(archive_path, split="student_test")
    if len(refs) != 10:
        raise ValueError(f"Expected 10 student-test trips; found {len(refs)}")
    artifact = joblib.load(artifact_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_seconds: dict[str, float] = {}

    # This pass opens exactly the three raw inference files, never .route.
    with ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="pathsnap_raw_") as temp_name:
        temp_dir = Path(temp_name)
        for ref in refs:
            saved_path = predictions_dir / f"{ref.trip_id}.npz"
            timing_path = predictions_dir / f"{ref.trip_id}.time.json"
            if saved_path.is_file() and timing_path.is_file():
                prediction_seconds[ref.trip_id] = json.loads(timing_path.read_text(encoding="utf-8"))["inference_seconds"]
                continue
            raw_paths = {}
            for suffix in (".track", ".nodes", ".arcs"):
                raw_path = temp_dir / f"{ref.trip_id}{suffix}"
                raw_path.write_bytes(archive.read(ref.files[suffix]))
                raw_paths[suffix] = raw_path
            start = time.perf_counter()
            detail = _predict_trip_details(raw_paths[".track"], raw_paths[".nodes"],
                                           raw_paths[".arcs"], artifact)
            elapsed = time.perf_counter() - start
            temporary_prediction = saved_path.with_suffix(".npz.tmp")
            with temporary_prediction.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    classifier_arc_ids=detail["classifier_arc_ids"],
                    decoded_arc_ids=detail["decoded_arc_ids"],
                    candidate_edge_ids=detail["candidate_edge_ids"],
                    candidate_point_indices=detail["candidate_point_indices"],
                    raw_probability=detail["raw_probability"],
                    calibrated_probability=detail["calibrated_probability"],
                    epsg=np.asarray(detail["epsg"]),
                )
            temporary_prediction.replace(saved_path)
            _atomic_json(timing_path, {"trip_id": ref.trip_id, "inference_seconds": elapsed,
                                       "point_count": detail["point_count"], "seed": seed})
            prediction_seconds[ref.trip_id] = elapsed
            del detail
            _atomic_json(progress_path, {
                "stage": "raw_predictions",
                "completed_trips": [ref.trip_id for ref in refs if (predictions_dir / f"{ref.trip_id}.npz").is_file()],
                "route_files_opened": False,
                "seed": seed,
            })
            print(f"[student raw inference] saved {ref.trip_id}", flush=True)

    # Only now open reference routes, assigning one route arc to each observation
    # using the same current-label heuristic that produced the training caches.
    per_trip: list[dict[str, Any]] = []
    raw_labels: list[np.ndarray] = []
    raw_probabilities: list[np.ndarray] = []
    calibrated_labels: list[np.ndarray] = []
    calibrated_probabilities: list[np.ndarray] = []
    for ref in refs:
        data = read_trip(archive_path, ref)
        from .feature_pipeline import build_trip_features
        labeled = build_trip_features(
            data, radius_m=float(artifact["candidate_radius_m"]),
            max_candidates=int(artifact["max_candidates"]),
            sample_spacing_m=float(artifact["sample_spacing_m"]),
            gap_reset_seconds=float(artifact["gap_reset_seconds"]), include_labels=True,
        )
        prediction_path = predictions_dir / f"{ref.trip_id}.npz"
        with np.load(prediction_path, allow_pickle=False) as saved:
            classifier_edges = saved["classifier_arc_ids"].copy()
            decoded_edges = saved["decoded_arc_ids"].copy()
            candidate_ids = saved["candidate_edge_ids"].copy()
            candidate_points = saved["candidate_point_indices"].copy()
            raw_probability = saved["raw_probability"].copy()
            calibrated_probability = saved["calibrated_probability"].copy()
        if (len(decoded_edges) != labeled.point_count
                or len(candidate_ids) != labeled.candidate_count
                or not np.array_equal(candidate_ids, labeled.candidate_edge_ids)
                or not np.array_equal(candidate_points, labeled.candidate_point_indices)):
            raise ValueError(f"Saved raw predictions do not align with {ref.trip_id}'s route labels")
        classifier_metric = _metric_for_edges(labeled, classifier_edges)
        hmm_metric = _metric_for_edges(labeled, decoded_edges)
        candidate_labels = (candidate_ids == labeled.point_true_edges[candidate_points]).astype(np.int8)
        per_trip.append({
            "trip_id": ref.trip_id,
            "observations": labeled.point_count,
            "candidate_miss_rate": classifier_metric["candidate_miss_rate"],
            "classifier_top1": classifier_metric["observation_top1_accuracy"],
            "classifier_route_continuity": classifier_metric["route_continuity"],
            "hmm_top1": hmm_metric["observation_top1_accuracy"],
            "hmm_route_continuity": hmm_metric["route_continuity"],
            "classifier_correct": classifier_metric["observation_correct"],
            "classifier_route_pairs_scored": classifier_metric["route_continuity_pairs_scored"],
            "classifier_connected": classifier_metric["connected_pairs"],
            "hmm_correct": hmm_metric["observation_correct"],
            "route_continuity_pairs_scored": hmm_metric["route_continuity_pairs_scored"],
            "hmm_connected": hmm_metric["connected_pairs"],
            "inference_seconds": prediction_seconds[ref.trip_id],
        })
        raw_labels.append(candidate_labels)
        calibrated_labels.append(candidate_labels)
        raw_probabilities.append(raw_probability)
        calibrated_probabilities.append(calibrated_probability)
        _atomic_json(output_dir / f"student_trip_{ref.trip_id}_metrics.json", per_trip[-1])
        _atomic_json(progress_path, {
            "stage": "route_scoring",
            "completed_trips": [row["trip_id"] for row in per_trip],
            "route_files_opened": True,
            "seed": seed,
        })
        del data, labeled, classifier_edges, decoded_edges, candidate_ids, candidate_points
        del raw_probability, calibrated_probability, candidate_labels

    from .robust_experiment import _observation_summary
    candidate_raw_summary = _candidate_summary(raw_labels, raw_probabilities)
    candidate_cal_summary = _candidate_summary(calibrated_labels, calibrated_probabilities)
    observations = sum(row["observations"] for row in per_trip)
    classifier_correct = sum(row["classifier_correct"] for row in per_trip)
    hmm_correct = sum(row["hmm_correct"] for row in per_trip)
    pairs = sum(row["route_continuity_pairs_scored"] for row in per_trip)
    classifier_pairs = sum(row["classifier_route_pairs_scored"] for row in per_trip)
    classifier_connected = sum(row["classifier_connected"] for row in per_trip)
    hmm_connected = sum(row["hmm_connected"] for row in per_trip)
    result = {
        "protocol": {
            "model": artifact["model_name"], "seed": seed,
            "training_and_selection_trips": 80,
            "student_test_trips_scored_once_after_freezing": 10,
            "inference_input": ".track, .nodes, .arcs only; route files opened only after all raw predictions were saved",
            "label_rule": "current monotone nearest route-arc assignment; labels were not changed",
        },
        "observations": observations,
        "candidate_miss_rate": sum(row["candidate_miss_rate"] * row["observations"] for row in per_trip) / max(1, observations),
        "classifier_only": {
            "pooled_top1": classifier_correct / max(1, observations),
            "mean_per_trip_top1": float(np.mean([row["classifier_top1"] for row in per_trip])),
            "route_continuity": classifier_connected / max(1, classifier_pairs),
            "per_trip": [{"trip_id": row["trip_id"], "top1": row["classifier_top1"]} for row in per_trip],
        },
        "classifier_plus_hmm": {
            "pooled_top1": hmm_correct / max(1, observations),
            "mean_per_trip_top1": float(np.mean([row["hmm_top1"] for row in per_trip])),
            "route_continuity": hmm_connected / max(1, pairs),
            "per_trip": [{"trip_id": row["trip_id"], "top1": row["hmm_top1"]} for row in per_trip],
            "trips_hmm_hurt": [row["trip_id"] for row in per_trip if row["hmm_top1"] < row["classifier_top1"]],
        },
        "candidate_macro_f1_and_logloss_raw": candidate_raw_summary,
        "candidate_macro_f1_and_logloss_platt": candidate_cal_summary,
        "per_trip_metrics": per_trip,
        "mean_inference_seconds_per_trip": float(np.mean(list(prediction_seconds.values()))),
        "artifact": str(artifact_path),
    }
    _atomic_json(final_path, result)
    _atomic_json(progress_path, {"stage": "complete", "completed_trips": [row["trip_id"] for row in per_trip],
                                 "route_files_opened": True, "seed": seed})
    with (output_dir / "student_test_per_trip.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["trip_id", "observations", "candidate_miss_rate", "classifier_top1",
                  "classifier_route_continuity", "hmm_top1", "hmm_route_continuity", "inference_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in per_trip)
    return result
