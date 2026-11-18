"""Nested trip-grouped comparison with training-only Platt calibration and HMM tuning.

This experiment uses the existing 15-feature caches and current labels. Student-test
caches are deliberately never opened here. Each outer-fold result and its held-out
predictions are written before the next fold starts.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .decoding import decode_trip_hmm
from .feature_pipeline import FEATURE_COLUMNS
from .feature_store import load_trip_feature_set
from .hmm_training import DECODER_GRID, _metric_for_classifier, _metric_for_edges
from .training import (
    _candidate_metrics,
    _fit_candidate_model,
    _observation_metrics,
    _prediction_map,
    _probability_for_model,
    _save_json,
    _training_matrix,
    model_zoo,
)

MODEL_NAMES = (
    "logistic_regression",
    "gaussian_naive_bayes",
    "random_forest",
    "hist_gradient_boosting",
    "knn",
)
LABEL_REVIEW_TRIPS = {"00000033", "00000054", "00000066"}
HIGH_MISS_TRIPS = {"00000002", "00000054", "00000066"}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def _predict_probabilities(model, name: str, data, sample_prior: float,
                           population_prior: float, chunk_rows: int = 100_000) -> np.ndarray:
    result = np.empty(data.candidate_count, dtype=np.float32)
    for start in range(0, data.candidate_count, chunk_rows):
        stop = min(data.candidate_count, start + chunk_rows)
        result[start:stop] = _probability_for_model(
            model, name, data.features[start:stop], sample_prior, population_prior)
    return result


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def _fit_platt(probabilities: list[np.ndarray], labels: list[np.ndarray], seed: int) -> dict[str, float]:
    raw = np.concatenate(probabilities)
    y = np.concatenate(labels).astype(np.int8, copy=False)
    calibrator = LogisticRegression(C=1.0, max_iter=100, solver="lbfgs", random_state=seed)
    calibrator.fit(_logit(raw).reshape(-1, 1), y)
    return {"slope": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
            "fit_rows": int(len(y)), "positive_rows": int(np.count_nonzero(y))}


def _apply_platt(probability: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    score = parameters["slope"] * _logit(probability) + parameters["intercept"]
    calibrated = np.empty_like(score)
    positive = score >= 0
    calibrated[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
    exp_score = np.exp(score[~positive])
    calibrated[~positive] = exp_score / (1.0 + exp_score)
    return np.clip(calibrated, 1e-7, 1 - 1e-7).astype(np.float32)


def _candidate_summary(labels: list[np.ndarray], probabilities: list[np.ndarray],
                       bins: int = 20) -> dict[str, Any]:
    y = np.concatenate(labels).astype(np.int8, copy=False)
    p = np.concatenate(probabilities).astype(np.float64, copy=False)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    metrics = _candidate_metrics(y, clipped)
    metrics["candidate_rows"] = int(len(y))
    metrics["positive_rows"] = int(np.count_nonzero(y))
    metrics["brier_score"] = float(np.mean((p - y) ** 2))
    metrics["mean_predicted_positive_probability"] = float(np.mean(p))
    bin_ids = np.minimum((p * bins).astype(np.int64), bins - 1)
    counts = np.bincount(bin_ids, minlength=bins)
    probability_sum = np.bincount(bin_ids, weights=p, minlength=bins)
    positive_sum = np.bincount(bin_ids, weights=y, minlength=bins)
    reliability = []
    ece = 0.0
    for index, count in enumerate(counts):
        if not count:
            continue
        mean_probability = float(probability_sum[index] / count)
        observed_rate = float(positive_sum[index] / count)
        ece += count / len(y) * abs(mean_probability - observed_rate)
        reliability.append({
            "bin_low": index / bins,
            "bin_high": (index + 1) / bins,
            "count": int(count),
            "mean_probability": mean_probability,
            "observed_positive_rate": observed_rate,
        })
    metrics["ece"] = float(ece)
    metrics["reliability_bins"] = reliability
    return metrics


def _observation_summary(trip_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not trip_rows:
        return {"observation_top1_accuracy": 0.0, "mean_per_trip_top1": 0.0,
                "candidate_miss_rate": 0.0, "prediction_coverage": 0.0,
                "route_continuity": 0.0, "observations_scored": 0}
    observations = sum(row["observations_scored"] for row in trip_rows)
    correct = sum(row["observation_correct"] for row in trip_rows)
    misses = sum(row["candidate_misses"] for row in trip_rows)
    covered = sum(row["predictions_covered"] for row in trip_rows)
    connected_pairs = sum(row["route_continuity_pairs_scored"] for row in trip_rows)
    connected = sum(row["connected_pairs"] for row in trip_rows)
    return {
        "observation_top1_accuracy": correct / max(1, observations),
        "mean_per_trip_top1": float(np.mean([row["observation_top1_accuracy"] for row in trip_rows])),
        "candidate_miss_rate": misses / max(1, observations),
        "prediction_coverage": covered / max(1, observations),
        "route_continuity": connected / max(1, connected_pairs),
        "observations_scored": int(observations),
        "candidate_misses": int(misses),
        "route_continuity_pairs_scored": int(connected_pairs),
    }


def _edge_predictions(data, probabilities: np.ndarray) -> np.ndarray:
    mapping = _prediction_map([data], {data.trip_id: probabilities})
    return np.asarray([mapping.get((data.trip_id, i), -1)
                       for i in range(data.point_count)], dtype=np.int64)


def _fit_record(fitted_model, model_name: str, seed: int,
                training_matrix: tuple[np.ndarray, np.ndarray, np.ndarray, float]):
    x, y, weights, population_prior = training_matrix
    model = clone(fitted_model)
    start = time.perf_counter()
    fitted, sample_prior = _fit_candidate_model(
        model, model_name, x, y, weights, population_prior, seed)
    return fitted, sample_prior, population_prior, time.perf_counter() - start


def _model_fold_result(model_name: str, fold_number: int, validation_trip_ids: list[str],
                       raw_probabilities: list[np.ndarray], calibrated_probabilities: list[np.ndarray],
                       candidate_labels: list[np.ndarray], raw_trip_metrics: list[dict],
                       calibrated_trip_metrics: list[dict], hmm_trip_metrics: list[dict],
                       hmm_params: dict[str, float], platt_parameters: dict[str, float],
                       timing: dict[str, float], seed: int) -> dict[str, Any]:
    def attach_trip_ids(rows: list[dict]) -> list[dict]:
        if len(rows) != len(validation_trip_ids):
            raise ValueError("Per-trip metric rows do not align with validation trip IDs")
        return [{"trip_id": trip_id, **row} for trip_id, row in zip(validation_trip_ids, rows)]

    return {
        "model": model_name,
        "outer_fold": fold_number,
        "seed": seed,
        "validation_trip_ids": validation_trip_ids,
        "classifier_raw": {
            "candidate_metrics": _candidate_summary(candidate_labels, raw_probabilities),
            "observation_metrics": _observation_summary(raw_trip_metrics),
            "per_trip": attach_trip_ids(raw_trip_metrics),
        },
        "classifier_platt_calibrated": {
            "calibrator": platt_parameters,
            "candidate_metrics": _candidate_summary(candidate_labels, calibrated_probabilities),
            "observation_metrics": _observation_summary(calibrated_trip_metrics),
            "per_trip": attach_trip_ids(calibrated_trip_metrics),
        },
        "classifier_plus_hmm": {
            "decoder_params": hmm_params,
            "observation_metrics": _observation_summary(hmm_trip_metrics),
            "per_trip": attach_trip_ids(hmm_trip_metrics),
        },
        "timing_seconds": timing,
        "feature_columns": list(FEATURE_COLUMNS),
    }


def _pooled_oof_summary(cache_dir: Path, output_dir: Path, model_name: str,
                        fold_by_trip: dict[str, int], exclude_ids: set[str] | None = None) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    paths = sorted((cache_dir / "train").glob("*.npz"))
    raw_labels: list[np.ndarray] = []
    calibrated_labels: list[np.ndarray] = []
    raw_probabilities: list[np.ndarray] = []
    calibrated_probabilities: list[np.ndarray] = []
    raw_trip_metrics: list[dict[str, Any]] = []
    calibrated_trip_metrics: list[dict[str, Any]] = []
    hmm_trip_metrics: list[dict[str, Any]] = []
    fold_top1 = {"raw": [], "calibrated": [], "hmm": []}
    per_trip_rows = []
    for path in paths:
        if path.stem in exclude_ids:
            continue
        data = load_trip_feature_set(path)
        prediction_path = output_dir / "predictions" / model_name / f"fold_{fold_by_trip[path.stem]}" / f"{path.stem}.npz"
        with np.load(prediction_path, allow_pickle=False) as saved:
            raw = saved["raw_probability"].copy()
            calibrated = saved["calibrated_probability"].copy()
            raw_edges = saved["raw_edges"].copy()
            calibrated_edges = saved["calibrated_edges"].copy()
            hmm_edges = saved["hmm_edges"].copy()
        labels = data.labels.astype(np.int8, copy=True)
        metric_raw = _observation_metrics([data], {(data.trip_id, i): int(edge)
            for i, edge in enumerate(raw_edges) if edge >= 0})
        metric_cal = _observation_metrics([data], {(data.trip_id, i): int(edge)
            for i, edge in enumerate(calibrated_edges) if edge >= 0})
        metric_hmm = _metric_for_edges(data, hmm_edges)
        raw_trip_metrics.append(metric_raw)
        calibrated_trip_metrics.append(metric_cal)
        hmm_trip_metrics.append(metric_hmm)
        fold_top1["raw"].append((fold_by_trip[path.stem], metric_raw["observation_top1_accuracy"]))
        fold_top1["calibrated"].append((fold_by_trip[path.stem], metric_cal["observation_top1_accuracy"]))
        fold_top1["hmm"].append((fold_by_trip[path.stem], metric_hmm["observation_top1_accuracy"]))
        per_trip_rows.append({
            "trip_id": data.trip_id,
            "fold": fold_by_trip[path.stem],
            "raw_top1": metric_raw["observation_top1_accuracy"],
            "calibrated_top1": metric_cal["observation_top1_accuracy"],
            "hmm_top1": metric_hmm["observation_top1_accuracy"],
            "candidate_miss_rate": metric_raw["candidate_miss_rate"],
            "raw_continuity": metric_raw["route_continuity"],
            "calibrated_continuity": metric_cal["route_continuity"],
            "hmm_continuity": metric_hmm["route_continuity"],
        })
        raw_labels.append(labels)
        calibrated_labels.append(labels)
        raw_probabilities.append(raw)
        calibrated_probabilities.append(calibrated)
        del data, labels, raw, calibrated, raw_edges, calibrated_edges, hmm_edges

    fold_means = {}
    for method in ("raw", "calibrated", "hmm"):
        fold_means[method] = {
            str(fold): float(np.mean([score for assigned, score in values if assigned == fold]))
            for fold in sorted(set(fold_by_trip.values()))
            for values in [fold_top1[method]]
            if any(assigned == fold for assigned, _ in values)
        }
    return {
        "classifier_raw": {
            "candidate_metrics": _candidate_summary(raw_labels, raw_probabilities),
            "observation_metrics": _observation_summary(raw_trip_metrics),
        },
        "classifier_platt_calibrated": {
            "candidate_metrics": _candidate_summary(calibrated_labels, calibrated_probabilities),
            "observation_metrics": _observation_summary(calibrated_trip_metrics),
        },
        "classifier_plus_hmm": {"observation_metrics": _observation_summary(hmm_trip_metrics)},
        "fold_mean_per_trip_top1": fold_means,
        "per_trip": per_trip_rows,
        "excluded_trip_ids": sorted(exclude_ids),
    }


def run_grouped_model_hmm_experiment(cache_dir: str | Path, output_dir: str | Path,
                                     n_splits: int = 5, seed: int = 42) -> dict[str, Any]:
    """Run nested GroupKFold comparisons, train-only calibration, and HMM tuning."""
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    random.seed(seed)
    np.random.seed(seed)
    train_paths = sorted((cache_dir / "train").glob("*.npz"))
    if len(train_paths) < n_splits:
        raise ValueError(f"Need at least {n_splits} training trips; found {len(train_paths)}")
    if not train_paths:
        raise ValueError(f"No training caches found at {cache_dir / 'train'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "experiment_progress.json"

    trip_ids = [path.stem for path in train_paths]
    outer_splits = list(GroupKFold(n_splits=n_splits).split(np.arange(len(trip_ids)), groups=trip_ids))
    fold_by_trip: dict[str, int] = {}
    fold_plan = []
    for fold_number, (_, validation_indices) in enumerate(outer_splits, start=1):
        valid_ids = [trip_ids[index] for index in validation_indices]
        fold_plan.append({"fold": fold_number, "validation_trip_ids": valid_ids})
        for trip_id in valid_ids:
            fold_by_trip[trip_id] = fold_number
    expected_fold_plan = {
        "protocol": "GroupKFold by train trip_id; identical outer folds for all five models",
        "seed": seed, "n_splits": n_splits, "training_trip_ids": trip_ids,
        "folds": fold_plan,
    }
    fold_plan_path = output_dir / "fold_plan.json"
    if fold_plan_path.exists():
        existing_plan = json.loads(fold_plan_path.read_text(encoding="utf-8"))
        if existing_plan != expected_fold_plan:
            raise ValueError(f"Saved fold plan does not match requested data or seed: {fold_plan_path}")
    else:
        _atomic_json(fold_plan_path, expected_fold_plan)

    expected_protocol = {
            "outer_cv": "GroupKFold by training trip_id",
            "inner_cv": "4-fold GroupKFold only within each outer training partition",
            "n_splits": n_splits,
            "seed": seed,
            "calibration": "Platt logistic regression fitted to inner OOF candidate probabilities from training trips only",
            "hmm_tuning": "Select decoder settings using inner OOF training-trip predictions only; outer validation trip IDs are excluded",
            "student_test_role": "not opened or used by this experiment; final one-time comparison only",
            "negative_sampling": "All positive rows plus at most 12 near/random negative rows per point, weighted to restore candidate prevalence",
            "model_names": list(MODEL_NAMES),
            "candidate_radius_m": 250.0,
            "max_candidates": 128,
            "feature_count": len(FEATURE_COLUMNS),
        }
    if state_path.exists():
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("protocol") != expected_protocol:
            raise ValueError(f"Saved experiment protocol does not match requested run: {state_path}")
        if (output_dir / "model_comparison_15feature_nested.json").is_file():
            return json.loads((output_dir / "model_comparison_15feature_nested.json").read_text(encoding="utf-8"))
    else:
        state = {
        "protocol": expected_protocol,
        "completed_outer_folds": 0,
        "completed_model_folds": {name: [] for name in MODEL_NAMES},
        "models": {name: {"folds": []} for name in MODEL_NAMES},
        }
        _atomic_json(state_path, state)
    zoo = model_zoo(seed)

    for outer_fold, (outer_train_indices, outer_valid_indices) in enumerate(outer_splits, start=1):
        if outer_fold <= int(state.get("completed_outer_folds", 0)):
            print(f"[outer fold {outer_fold}/{n_splits}] already saved; resuming after it", flush=True)
            continue
        outer_train_paths = [train_paths[index] for index in outer_train_indices]
        outer_valid_paths = [train_paths[index] for index in outer_valid_indices]
        inner_ids = [path.stem for path in outer_train_paths]
        inner_splits = list(GroupKFold(n_splits=min(4, len(inner_ids))).split(
            np.arange(len(inner_ids)), groups=inner_ids))
        inner_raw: dict[str, dict[str, np.ndarray]] = {name: {} for name in MODEL_NAMES}
        inner_labels: dict[str, np.ndarray] = {}
        inner_fit_seconds: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}
        inner_matrix_seconds = 0.0
        inner_inference_seconds: dict[str, float] = {name: 0.0 for name in MODEL_NAMES}

        print(f"[outer fold {outer_fold}/{n_splits}] inner OOF fits", flush=True)
        for inner_fold, (inner_train_indices, inner_valid_indices) in enumerate(inner_splits, start=1):
            inner_train_paths = [outer_train_paths[index] for index in inner_train_indices]
            inner_valid_paths = [outer_train_paths[index] for index in inner_valid_indices]
            matrix_start = time.perf_counter()
            matrix = _training_matrix(inner_train_paths, seed + outer_fold * 100 + inner_fold)
            inner_matrix_seconds += time.perf_counter() - matrix_start
            fitted_models: dict[str, tuple[Any, float, float]] = {}
            model_seed = seed + outer_fold * 1000 + inner_fold * 10
            for model_index, name in enumerate(MODEL_NAMES):
                estimator = clone(model_zoo(model_seed + model_index)[name])
                fitted, sample_prior, population_prior, fit_seconds = _fit_record(
                    estimator, name, model_seed + model_index, matrix)
                inner_fit_seconds[name].append(fit_seconds)
                fitted_models[name] = (fitted, sample_prior, population_prior)
            for path in inner_valid_paths:
                data = load_trip_feature_set(path)
                inner_labels[data.trip_id] = data.labels.astype(np.int8, copy=True)
                for name in MODEL_NAMES:
                    fitted, sample_prior, population_prior = fitted_models[name]
                    start = time.perf_counter()
                    probability = _predict_probabilities(fitted, name, data, sample_prior, population_prior)
                    inner_inference_seconds[name] += time.perf_counter() - start
                    inner_raw[name][data.trip_id] = probability
                del data
            del fitted_models, matrix
            print(f"[outer fold {outer_fold}/{n_splits}] inner fold {inner_fold}/{len(inner_splits)} saved in memory", flush=True)

        calibrators: dict[str, dict[str, float]] = {}
        calibration_seconds: dict[str, float] = {}
        for name in MODEL_NAMES:
            ordered_ids = [path.stem for path in outer_train_paths]
            start = time.perf_counter()
            calibrators[name] = _fit_platt(
                [inner_raw[name][trip_id] for trip_id in ordered_ids],
                [inner_labels[trip_id] for trip_id in ordered_ids],
                seed + outer_fold * 1000 + MODEL_NAMES.index(name))
            calibration_seconds[name] = time.perf_counter() - start

        tune_scores: dict[str, dict[str, list[dict[str, Any]]]] = {
            name: {json.dumps(params, sort_keys=True): [] for params in DECODER_GRID}
            for name in MODEL_NAMES
        }
        tune_params = {json.dumps(params, sort_keys=True): params for params in DECODER_GRID}
        hmm_tuning_seconds: dict[str, float] = {name: 0.0 for name in MODEL_NAMES}
        for path in outer_train_paths:
            data = load_trip_feature_set(path)
            for name in MODEL_NAMES:
                calibrated = _apply_platt(inner_raw[name][data.trip_id], calibrators[name])
                for key, params in tune_params.items():
                    start = time.perf_counter()
                    decoded = decode_trip_hmm(data, calibrated, **params)
                    hmm_tuning_seconds[name] += time.perf_counter() - start
                    tune_scores[name][key].append(_metric_for_edges(data, decoded))
            del data

        selected_hmm: dict[str, dict[str, float]] = {}
        selected_hmm_scores: dict[str, float] = {}
        for name in MODEL_NAMES:
            scored = []
            for key, trip_metrics in tune_scores[name].items():
                total = _observation_summary(trip_metrics)["observation_top1_accuracy"]
                scored.append((total, key))
            best_score, best_key = max(scored, key=lambda item: (item[0], item[1]))
            selected_hmm[name] = tune_params[best_key]
            selected_hmm_scores[name] = float(best_score)

        outer_matrix_start = time.perf_counter()
        outer_matrix = _training_matrix(outer_train_paths, seed + outer_fold * 10000)
        outer_matrix_seconds = time.perf_counter() - outer_matrix_start
        fitted_outer: dict[str, tuple[Any, float, float, float]] = {}
        outer_fit_seconds: dict[str, float] = {}
        outer_model_seed = seed + outer_fold * 10000
        for model_index, name in enumerate(MODEL_NAMES):
            estimator = clone(model_zoo(outer_model_seed + model_index)[name])
            fitted, sample_prior, population_prior, fit_seconds = _fit_record(
                estimator, name, outer_model_seed + model_index, outer_matrix)
            fitted_outer[name] = (fitted, sample_prior, population_prior, fit_seconds)
            outer_fit_seconds[name] = fit_seconds
        del outer_matrix

        fold_payloads: dict[str, dict[str, Any]] = {}
        fold_arrays: dict[str, dict[str, list[np.ndarray]]] = {
            name: {"raw": [], "calibrated": [], "labels": []} for name in MODEL_NAMES
        }
        fold_trip_metrics: dict[str, dict[str, list[dict[str, Any]]]] = {
            name: {"raw": [], "calibrated": [], "hmm": []} for name in MODEL_NAMES
        }
        validation_inference_seconds = {name: 0.0 for name in MODEL_NAMES}
        validation_hmm_seconds = {name: 0.0 for name in MODEL_NAMES}
        validation_trip_ids = []

        for path in outer_valid_paths:
            data = load_trip_feature_set(path)
            validation_trip_ids.append(data.trip_id)
            for name in MODEL_NAMES:
                fitted, sample_prior, population_prior, _ = fitted_outer[name]
                start = time.perf_counter()
                raw_probability = _predict_probabilities(fitted, name, data, sample_prior, population_prior)
                validation_inference_seconds[name] += time.perf_counter() - start
                calibrated_probability = _apply_platt(raw_probability, calibrators[name])
                raw_metric = _metric_for_classifier(data, raw_probability)
                calibrated_metric = _metric_for_classifier(data, calibrated_probability)
                start = time.perf_counter()
                hmm_edges = decode_trip_hmm(data, calibrated_probability, **selected_hmm[name])
                validation_hmm_seconds[name] += time.perf_counter() - start
                hmm_metric = _metric_for_edges(data, hmm_edges)
                fold_trip_metrics[name]["raw"].append(raw_metric)
                fold_trip_metrics[name]["calibrated"].append(calibrated_metric)
                fold_trip_metrics[name]["hmm"].append(hmm_metric)
                fold_arrays[name]["raw"].append(raw_probability)
                fold_arrays[name]["calibrated"].append(calibrated_probability)
                fold_arrays[name]["labels"].append(data.labels.astype(np.int8, copy=True))
                prediction_dir = output_dir / "predictions" / name / f"fold_{outer_fold}"
                prediction_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    prediction_dir / f"{data.trip_id}.npz",
                    raw_probability=raw_probability,
                    calibrated_probability=calibrated_probability,
                    raw_edges=_edge_predictions(data, raw_probability),
                    calibrated_edges=_edge_predictions(data, calibrated_probability),
                    hmm_edges=hmm_edges,
                )
            del data

        for name in MODEL_NAMES:
            timing = {
                "inner_training_matrix_seconds_shared": float(inner_matrix_seconds),
                "inner_fit_seconds_total": float(sum(inner_fit_seconds[name])),
                "inner_fit_seconds_by_fold": [float(value) for value in inner_fit_seconds[name]],
                "inner_validation_inference_seconds": float(inner_inference_seconds[name]),
                "platt_calibration_fit_seconds": float(calibration_seconds[name]),
                "inner_hmm_selection_seconds": float(hmm_tuning_seconds[name]),
                "outer_training_matrix_seconds_shared": float(outer_matrix_seconds),
                "outer_model_fit_seconds": float(outer_fit_seconds[name]),
                "outer_validation_inference_seconds": float(validation_inference_seconds[name]),
                "outer_hmm_decode_seconds": float(validation_hmm_seconds[name]),
                "training_seconds_total": float(inner_matrix_seconds + sum(inner_fit_seconds[name]) +
                    inner_inference_seconds[name] + calibration_seconds[name] + hmm_tuning_seconds[name] +
                    outer_matrix_seconds + outer_fit_seconds[name]),
            }
            result = _model_fold_result(
                name, outer_fold, validation_trip_ids,
                fold_arrays[name]["raw"], fold_arrays[name]["calibrated"], fold_arrays[name]["labels"],
                fold_trip_metrics[name]["raw"], fold_trip_metrics[name]["calibrated"],
                fold_trip_metrics[name]["hmm"], selected_hmm[name], calibrators[name], timing,
                outer_model_seed + MODEL_NAMES.index(name),
            )
            model_dir = output_dir / "folds" / name
            _atomic_json(model_dir / f"fold_{outer_fold}.json", result)
            state["models"][name]["folds"] = [
                previous for previous in state["models"][name]["folds"]
                if int(previous.get("fold", -1)) != outer_fold
            ]
            state["models"][name]["folds"].append({
                "fold": outer_fold,
                "classifier_raw": result["classifier_raw"]["observation_metrics"],
                "classifier_platt_calibrated": result["classifier_platt_calibrated"]["observation_metrics"],
                "classifier_plus_hmm": result["classifier_plus_hmm"]["observation_metrics"],
                "candidate_raw": result["classifier_raw"]["candidate_metrics"],
                "candidate_calibrated": result["classifier_platt_calibrated"]["candidate_metrics"],
                "hmm_params": selected_hmm[name],
                "hmm_inner_training_top1": selected_hmm_scores[name],
                "timing_seconds": timing,
            })
            state["completed_model_folds"][name] = sorted(set(
                state["completed_model_folds"][name] + [outer_fold]))
            _atomic_json(state_path, state)
            print(f"[outer fold {outer_fold}/{n_splits}] saved {name}", flush=True)
            fold_payloads[name] = result

        state["completed_outer_folds"] = outer_fold
        _atomic_json(state_path, state)
        del fitted_outer, fold_arrays, fold_trip_metrics, fold_payloads

    for name in MODEL_NAMES:
        state["models"][name]["all_training_oof"] = _pooled_oof_summary(
            cache_dir, output_dir, name, fold_by_trip)
        state["models"][name]["excluding_label_review_trips"] = _pooled_oof_summary(
            cache_dir, output_dir, name, fold_by_trip, LABEL_REVIEW_TRIPS)
        state["models"][name]["excluding_high_miss_trips"] = _pooled_oof_summary(
            cache_dir, output_dir, name, fold_by_trip, HIGH_MISS_TRIPS)

    state["selected_model_by_hmm_mean_per_trip_top1"] = max(
        MODEL_NAMES,
        key=lambda name: state["models"][name]["all_training_oof"]["classifier_plus_hmm"]
            ["observation_metrics"]["mean_per_trip_top1"],
    )
    _atomic_json(output_dir / "model_comparison_15feature_nested.json", state)
    _atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/interim/features")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_grouped_model_hmm_experiment(args.cache_dir, args.output_dir,
                                             n_splits=args.folds, seed=args.seed)
    brief = {name: {
        "raw_pooled_top1": result["models"][name]["all_training_oof"]["classifier_raw"]["observation_metrics"]["observation_top1_accuracy"],
        "calibrated_pooled_top1": result["models"][name]["all_training_oof"]["classifier_platt_calibrated"]["observation_metrics"]["observation_top1_accuracy"],
        "hmm_pooled_top1": result["models"][name]["all_training_oof"]["classifier_plus_hmm"]["observation_metrics"]["observation_top1_accuracy"],
        "hmm_mean_trip_top1": result["models"][name]["all_training_oof"]["classifier_plus_hmm"]["observation_metrics"]["mean_per_trip_top1"],
    } for name in MODEL_NAMES}
    print(json.dumps({"selected_model": result["selected_model_by_hmm_mean_per_trip_top1"],
                      "models": brief}, indent=2))


if __name__ == "__main__":
    main()
