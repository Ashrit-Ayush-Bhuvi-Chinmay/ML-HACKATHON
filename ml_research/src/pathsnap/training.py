"""Group-aware model comparison with candidate, observation, and route metrics."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .feature_pipeline import FEATURE_COLUMNS, TripFeatureSet


def model_zoo(seed: int = 42) -> dict[str, Any]:
    """Five distinct algorithm families; all expose candidate-match probabilities."""
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=400, random_state=seed)),
        "gaussian_naive_bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
            random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08,
            max_leaf_nodes=31, l2_regularization=0.1, random_state=seed),
        # Three metric features and a KD-tree keep this baseline tractable at PathSnap scale.
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11,
            weights="distance", algorithm="kd_tree", leaf_size=40, n_jobs=-1)),
    }


def flatten_sets(sets: Iterable[TripFeatureSet]) -> dict[str, np.ndarray]:
    sets = list(sets)
    populated = [s for s in sets if s.candidate_count]
    if not populated:
        raise ValueError("No point-to-arc candidate rows were generated.")
    return {
        "X": np.concatenate([s.features for s in populated], axis=0),
        "y": np.concatenate([s.labels for s in populated], axis=0),
        "groups": np.concatenate([np.full(s.candidate_count, s.trip_id, dtype=object) for s in populated]),
        "trip_sets": populated,
    }


def _positive_probability(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.classes_ if hasattr(model, "classes_") else model[-1].classes_)
    positive = classes.index(1)
    return np.asarray(proba[:, positive], dtype=np.float64)


def _candidate_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "candidate_accuracy": float(accuracy_score(y_true, prediction)),
        "candidate_macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "candidate_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "candidate_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "candidate_log_loss": float(log_loss(y_true, np.column_stack([1-probability, probability]), labels=[0, 1])),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def _prediction_map(sets: list[TripFeatureSet], probabilities_by_set: dict[str, np.ndarray]) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for data in sets:
        probabilities = probabilities_by_set.get(data.trip_id)
        if probabilities is None:
            continue
        for point_idx in range(data.point_count):
            start, stop = int(data.point_offsets[point_idx]), int(data.point_offsets[point_idx+1])
            if start == stop:
                continue
            local = probabilities[start:stop]
            # Candidate order is distance-rank order, so argmax preserves that tie-break.
            result[(data.trip_id, point_idx)] = int(data.candidate_edge_ids[start + int(np.argmax(local))])
    return result


def _observation_metrics(sets: list[TripFeatureSet], prediction_map: dict[tuple[str, int], int]) -> dict[str, Any]:
    correct = total = misses = 0
    predicted_coverage = predicted_count = 0
    connected = continuity_pairs = 0
    by_trip = {s.trip_id: s for s in sets}
    for data in sets:
        predicted_sequence = []
        for i, true_edge in enumerate(data.point_true_edges):
            total += 1
            misses += int(data.point_candidate_misses[i])
            pred = prediction_map.get((data.trip_id, i))
            predicted_sequence.append(pred)
            if pred is not None:
                predicted_count += 1
                correct += int(pred == int(true_edge))
        for i in range(1, data.point_count):
            if data.point_gap_resets[i]:
                continue
            previous, current = predicted_sequence[i-1], predicted_sequence[i]
            if previous is None or current is None:
                continue
            continuity_pairs += 1
            if previous == current or int(data.arcs[previous, 1]) == int(data.arcs[current, 0]):
                connected += 1
    return {
        "observation_top1_accuracy": correct / total if total else 0.0,
        "candidate_miss_rate": misses / total if total else 0.0,
        "prediction_coverage": predicted_count / total if total else 0.0,
        "route_continuity": connected / continuity_pairs if continuity_pairs else 0.0,
        "route_continuity_pairs_scored": continuity_pairs,
        "observation_correct": correct, "observations_scored": total,
        "candidate_misses": misses, "predictions_covered": predicted_count,
        "connected_pairs": connected,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return repr(value)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def compare_models(train_sets: Iterable[TripFeatureSet], student_sets: Iterable[TripFeatureSet],
                   output_dir: str | Path, n_splits: int = 5, seed: int = 42,
                   model_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Run GroupKFold CV on training trips, then one student-test comparison per model.

    The student-test labels are never included in GroupKFold, fitting, or hyperparameter search.
    Results are written after each model finishes so interrupted experiments remain auditable.
    """
    train_sets, student_sets = list(train_sets), list(student_sets)
    train = flatten_sets(train_sets)
    student = flatten_sets(student_sets)
    zoo = model_zoo(seed)
    names = list(model_names or zoo.keys())
    missing = set(names) - set(zoo)
    if missing:
        raise ValueError(f"Unknown models: {sorted(missing)}")
    if len(np.unique(train["y"])) < 2:
        raise ValueError("Training candidate labels contain only one class.")
    folds = list(GroupKFold(n_splits=n_splits).split(train["X"], train["y"], train["groups"]))
    root = Path(output_dir)
    cv_dir, student_dir = root / "cv", root / "student_test"
    cv_dir.mkdir(parents=True, exist_ok=True)
    student_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"protocol": {"cv": "GroupKFold grouped by train trip_id",
                                              "n_splits": n_splits, "seed": seed,
                                              "selection_metric": "mean observation_top1_accuracy",
                                              "student_test_role": "development-stage comparison only"},
                               "feature_columns": list(FEATURE_COLUMNS), "models": {}}
    summary_path = root / "model_comparison.json"
    csv_path = root / "model_comparison.csv"
    for name in names:
        fold_metrics = []
        for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
            model = clone(zoo[name])
            start_fit = time.perf_counter()
            model.fit(train["X"][train_idx], train["y"][train_idx])
            train_seconds = time.perf_counter() - start_fit
            start_predict = time.perf_counter()
            probabilities = _positive_probability(model, train["X"][valid_idx])
            predict_seconds = time.perf_counter() - start_predict
            valid_trip_ids = set(train["groups"][valid_idx].tolist())
            valid_sets = [s for s in train_sets if s.trip_id in valid_trip_ids]
            per_trip_probabilities = {}
            for trip_id in valid_trip_ids:
                per_trip_probabilities[trip_id] = np.zeros(next(s.candidate_count for s in valid_sets if s.trip_id == trip_id), dtype=np.float64)
            # Restore each held-out trip's compact row order.
            valid_by_trip: dict[str, list[tuple[int, float]]] = {}
            for idx, probability in zip(valid_idx, probabilities):
                valid_by_trip.setdefault(str(train["groups"][idx]), []).append((int(idx), float(probability)))
            offset_by_trip = {}
            for data in valid_sets:
                global_rows = np.flatnonzero(train["groups"] == data.trip_id)
                offset_by_trip[data.trip_id] = {int(g): i for i, g in enumerate(global_rows)}
            for trip_id, values in valid_by_trip.items():
                per_trip_probabilities[trip_id] = np.asarray([p for _, p in sorted(values, key=lambda x: offset_by_trip[trip_id][x[0]])])
            prediction_map = _prediction_map(valid_sets, per_trip_probabilities)
            metrics = _candidate_metrics(train["y"][valid_idx], probabilities)
            metrics.update(_observation_metrics(valid_sets, prediction_map))
            metrics.update({"fold": fold_index, "train_seconds": train_seconds,
                            "inference_seconds": predict_seconds,
                            "inference_ms_per_observation": 1000 * predict_seconds / max(1, sum(s.point_count for s in valid_sets))})
            fold_metrics.append(metrics)
            _save_json(cv_dir / f"{name}.json", {"model": name, "folds_completed": fold_index,
                                                    "fold_metrics": fold_metrics,
                                                    "configuration": _json_safe(zoo[name].get_params(deep=False))})
        metric_keys = ["candidate_macro_f1", "candidate_log_loss", "observation_top1_accuracy",
                       "candidate_miss_rate", "route_continuity", "train_seconds", "inference_seconds"]
        cv_summary = {f"mean_{k}": float(np.mean([f[k] for f in fold_metrics])) for k in metric_keys}
        cv_summary.update({f"std_{k}": float(np.std([f[k] for f in fold_metrics])) for k in metric_keys})

        full_model = clone(zoo[name])
        start_fit = time.perf_counter()
        full_model.fit(train["X"], train["y"])
        student_train_seconds = time.perf_counter() - start_fit
        start_predict = time.perf_counter()
        student_probabilities = _positive_probability(full_model, student["X"])
        student_inference_seconds = time.perf_counter() - start_predict
        student_prob_by_trip = {}
        cursor = 0
        for data in student_sets:
            student_prob_by_trip[data.trip_id] = student_probabilities[cursor:cursor+data.candidate_count]
            cursor += data.candidate_count
        student_prediction_map = _prediction_map(student_sets, student_prob_by_trip)
        student_metrics = _candidate_metrics(student["y"], student_probabilities)
        student_metrics.update(_observation_metrics(student_sets, student_prediction_map))
        student_metrics.update({"train_seconds": student_train_seconds,
                                "inference_seconds": student_inference_seconds,
                                "inference_ms_per_observation": 1000*student_inference_seconds/max(1, sum(s.point_count for s in student_sets))})
        result = {"model": name, "configuration": _json_safe(zoo[name].get_params(deep=False)),
                  "cv": cv_summary, "cv_folds": fold_metrics, "student_test": student_metrics}
        summary["models"][name] = result
        _save_json(student_dir / f"{name}.json", result)
        _save_json(summary_path, summary)
        _write_summary_csv(csv_path, summary)
    summary["selected_by_cv_top1"] = max(names, key=lambda n: summary["models"][n]["cv"]["mean_observation_top1_accuracy"])
    _save_json(summary_path, summary)
    _write_summary_csv(csv_path, summary)
    return summary


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for name, result in summary["models"].items():
        row = {"model": name, "algorithm_family": _family_name(name)}
        row.update(result["cv"])
        if "student_test" in result:
            row.update({f"student_{k}": v for k, v in result["student_test"].items()})
        rows.append(row)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _family_name(name: str) -> str:
    return {"logistic_regression": "linear/probabilistic", "gaussian_naive_bayes": "probabilistic",
            "random_forest": "bagged trees", "hist_gradient_boosting": "boosting",
            "knn": "instance-based"}.get(name, "other")


def _observation_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("observation_correct", "observations_scored", "candidate_misses",
            "predictions_covered", "connected_pairs", "route_continuity_pairs_scored")
    sums = {key: sum(row[key] for row in rows) for key in keys}
    return {"observation_top1_accuracy": sums["observation_correct"] / max(1, sums["observations_scored"]),
            "candidate_miss_rate": sums["candidate_misses"] / max(1, sums["observations_scored"]),
            "prediction_coverage": sums["predictions_covered"] / max(1, sums["observations_scored"]),
            "route_continuity": sums["connected_pairs"] / max(1, sums["route_continuity_pairs_scored"]),
            "observations_scored": sums["observations_scored"],
            "candidate_misses": sums["candidate_misses"],
            "route_continuity_pairs_scored": sums["route_continuity_pairs_scored"]}


def evaluate_nearest_arc_baseline(train_paths, student_paths, output_dir: str | Path,
                                  n_splits: int = 5, seed: int = 42) -> dict[str, Any]:
    """Score the closest candidate with the same observation-level Top-1 metric as models."""
    from .feature_store import load_trip_feature_set
    train_paths, student_paths = list(map(Path, train_paths)), list(map(Path, student_paths))
    trip_metrics = {}
    for split, paths in (("train", train_paths), ("student_test", student_paths)):
        for path in paths:
            data = load_trip_feature_set(path)
            predictions = {}
            for point_idx in range(data.point_count):
                start, stop = int(data.point_offsets[point_idx]), int(data.point_offsets[point_idx + 1])
                if start != stop:
                    nearest = start + int(np.argmin(data.candidate_distances_m[start:stop]))
                    predictions[(data.trip_id, point_idx)] = int(data.candidate_edge_ids[nearest])
            metrics = _observation_metrics([data], predictions)
            metrics["trip_id"] = data.trip_id
            metrics["split"] = split
            trip_metrics[(split, data.trip_id)] = metrics
            del data
    splitter = GroupKFold(n_splits=n_splits)
    train_ids = [path.stem for path in train_paths]
    folds = []
    for fold_no, (_, valid_idx) in enumerate(splitter.split(np.arange(len(train_paths)), groups=train_ids), start=1):
        rows = [trip_metrics[("train", train_ids[i])] for i in valid_idx]
        folds.append({"fold": fold_no, **_observation_metric_summary(rows)})
    train_rows = [trip_metrics[("train", trip_id)] for trip_id in train_ids]
    outlier_ids = [row["trip_id"] for row in sorted(train_rows,
        key=lambda row: (-row["candidate_miss_rate"], row["trip_id"]))[:3]]
    train_without_outliers = [row for row in train_rows if row["trip_id"] not in outlier_ids]
    student_rows = [trip_metrics[("student_test", path.stem)] for path in student_paths]
    fold_top1 = [row["observation_top1_accuracy"] for row in folds]
    result = {"baseline": "nearest candidate by projected point-to-arc distance",
        "selection_or_fit": "none; deterministic geometric baseline",
        "training_oof": {**_observation_metric_summary(train_rows),
            "fold_top1_mean": float(np.mean(fold_top1)), "fold_top1_std": float(np.std(fold_top1)),
            "folds": folds, "without_three_highest_miss_trips":
                _observation_metric_summary(train_without_outliers),
            "excluded_trip_ids": outlier_ids},
        "student_test_development_comparison": _observation_metric_summary(student_rows),
        "trip_metrics": list(trip_metrics.values())}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _save_json(target / "nearest_arc_baseline.json", result)
    with (target / "nearest_arc_baseline_trips.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "trip_id", "observation_top1_accuracy", "candidate_miss_rate",
                  "prediction_coverage", "route_continuity", "observations_scored"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in trip_metrics.values():
            writer.writerow({field: row.get(field) for field in fields})
    return result


def _stratified_point_sample(data: TripFeatureSet, rng: np.random.Generator,
                             max_negatives_per_point: int = 12,
                             nearest_negatives_per_point: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retain each positive plus near/random negatives; weights restore candidate prevalence."""
    selected_parts, weight_parts = [], []
    for point_idx in range(data.point_count):
        start, stop = int(data.point_offsets[point_idx]), int(data.point_offsets[point_idx + 1])
        if start == stop:
            continue
        local = np.arange(start, stop, dtype=np.int64)
        positives = local[data.labels[start:stop] == 1]
        negatives = local[data.labels[start:stop] == 0]
        keep = min(max_negatives_per_point, len(negatives))
        nearest = min(nearest_negatives_per_point, keep)
        chosen_neg = negatives[:nearest]
        extra = keep - nearest
        if extra:
            remaining = negatives[nearest:]
            chosen_neg = np.concatenate((chosen_neg, rng.choice(remaining, extra, replace=False)
                                         if len(remaining) > extra else remaining))
        chosen = np.concatenate((positives, chosen_neg))
        if len(chosen):
            weights = np.ones(len(chosen), dtype=np.float32)
            if len(chosen_neg):
                weights[len(positives):] = len(negatives) / len(chosen_neg)
            selected_parts.append(chosen)
            weight_parts.append(weights)
    if not selected_parts:
        raise ValueError(f"Trip {data.trip_id} has no labeled candidate rows")
    indices = np.concatenate(selected_parts)
    return indices, data.labels[indices], np.concatenate(weight_parts)


def _training_matrix(paths, seed: int, max_negatives_per_point: int = 12):
    from .feature_store import load_trip_feature_set
    xs, ys, ws = [], [], []
    candidate_total = positive_total = 0
    for path in paths:
        data = load_trip_feature_set(path)
        rng = np.random.default_rng(seed + sum(map(ord, data.trip_id)))
        rows, labels, weights = _stratified_point_sample(data, rng, max_negatives_per_point)
        xs.append(data.features[rows])
        ys.append(labels)
        ws.append(weights)
        candidate_total += data.candidate_count
        positive_total += int(np.count_nonzero(data.labels == 1))
        del data
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ws),
            positive_total / max(1, candidate_total))


def _fit_candidate_model(model, name: str, x: np.ndarray, y: np.ndarray,
                         weights: np.ndarray, population_prior: float, seed: int):
    """Fit prevalence-weighted models; cap and prior-correct the KNN reference set."""
    if name == "knn":
        rng = np.random.default_rng(seed)
        positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        limit = 50_000
        pos_count = min(len(positives), limit // 4)
        neg_count = min(len(negatives), limit - pos_count)
        pos_rows = rng.choice(positives, pos_count, replace=False) if pos_count else np.empty(0, dtype=int)
        neg_rows = rng.choice(negatives, neg_count, replace=False) if neg_count else np.empty(0, dtype=int)
        rows = np.concatenate((pos_rows, neg_rows))
        model.fit(x[rows][:, [0, 3, 10]], y[rows])
        return model, float(np.mean(y[rows])) if len(rows) else population_prior
    if hasattr(model, "steps"):
        final_name = model.steps[-1][0]
        model.fit(x, y, **{f"{final_name}__sample_weight": weights})
    else:
        model.fit(x, y, sample_weight=weights)
    return model, population_prior


def _probability_for_model(model, name: str, x: np.ndarray,
                           sample_prior: float, population_prior: float) -> np.ndarray:
    model_x = x[:, [0, 3, 10]] if name == "knn" else x
    probability = _positive_probability(model, model_x)
    if name == "knn" and 0 < sample_prior < 1 and 0 < population_prior < 1:
        ratio = (population_prior / (1 - population_prior)) / (sample_prior / (1 - sample_prior))
        probability = probability * ratio / (1 - probability + probability * ratio)
    return np.clip(probability, 1e-7, 1 - 1e-7)


def _score_trip(model, name: str, data: TripFeatureSet, sample_prior: float,
                population_prior: float, chunk_rows: int = 100_000) -> dict[str, Any]:
    from .feature_store import load_trip_feature_set  # noqa: F401
    probabilities = np.empty(data.candidate_count, dtype=np.float32)
    for start in range(0, data.candidate_count, chunk_rows):
        stop = min(data.candidate_count, start + chunk_rows)
        probabilities[start:stop] = _probability_for_model(
            model, name, data.features[start:stop], sample_prior, population_prior)
    labels = data.labels
    positive, predicted = labels == 1, probabilities >= 0.5
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    loss = float(-np.sum(labels * np.log(probabilities) +
                         (1-labels) * np.log1p(-probabilities), dtype=np.float64))
    point_predictions = np.full(data.point_count, -1, dtype=np.int64)
    correct = covered = 0
    for i, true_edge in enumerate(data.point_true_edges):
        start, stop = int(data.point_offsets[i]), int(data.point_offsets[i + 1])
        if start == stop:
            continue
        best = start + int(np.argmax(probabilities[start:stop]))
        pred_edge = int(data.candidate_edge_ids[best])
        point_predictions[i] = pred_edge
        covered += 1
        correct += int(pred_edge == int(true_edge))
    continuity_pairs = connected = 0
    for i in range(1, data.point_count):
        if data.point_gap_resets[i] or point_predictions[i-1] < 0 or point_predictions[i] < 0:
            continue
        previous, current = point_predictions[i-1], point_predictions[i]
        continuity_pairs += 1
        if previous == current or int(data.arcs[previous, 1]) == int(data.arcs[current, 0]):
            connected += 1
    return {"candidate_rows": data.candidate_count, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "candidate_loss_sum": loss, "point_total": data.point_count, "point_correct": correct,
            "candidate_misses": int(np.count_nonzero(data.point_candidate_misses)), "covered": covered,
            "continuity_pairs": continuity_pairs, "connected_pairs": connected}


def _merge_score_stats(stats: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("candidate_rows", "tp", "fp", "fn", "tn", "candidate_loss_sum", "point_total",
            "point_correct", "candidate_misses", "covered", "continuity_pairs", "connected_pairs")
    sums = {key: sum(row[key] for row in stats) for key in keys}
    tp, fp, fn, tn = sums["tp"], sums["fp"], sums["fn"], sums["tn"]
    f1_pos = 2 * tp / max(1, 2 * tp + fp + fn)
    f1_neg = 2 * tn / max(1, 2 * tn + fp + fn)
    return {"candidate_accuracy": (tp + tn) / max(1, sums["candidate_rows"]),
            "candidate_macro_f1": (f1_pos + f1_neg) / 2,
            "candidate_precision": tp / max(1, tp + fp), "candidate_recall": tp / max(1, tp + fn),
            "candidate_log_loss": sums["candidate_loss_sum"] / max(1, sums["candidate_rows"]),
            "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
            "observation_top1_accuracy": sums["point_correct"] / max(1, sums["point_total"]),
            "candidate_miss_rate": sums["candidate_misses"] / max(1, sums["point_total"]),
            "prediction_coverage": sums["covered"] / max(1, sums["point_total"]),
            "route_continuity": sums["connected_pairs"] / max(1, sums["continuity_pairs"]),
            "route_continuity_pairs_scored": sums["continuity_pairs"],
            "candidate_rows_scored": sums["candidate_rows"], "observations_scored": sums["point_total"]}


def compare_cached_models(train_paths, student_paths, output_dir: str | Path,
                          n_splits: int = 5, seed: int = 42,
                          model_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """GroupKFold by train trip; score all validation candidates using bounded memory."""
    from .feature_store import load_trip_feature_set
    import joblib
    train_paths, student_paths = list(map(Path, train_paths)), list(map(Path, student_paths))
    if not train_paths or not student_paths:
        raise ValueError("Train and student feature caches are both required")
    splitter = GroupKFold(n_splits=n_splits)
    folds = list(splitter.split(np.arange(len(train_paths)), groups=[p.stem for p in train_paths]))
    zoo = model_zoo(seed)
    names = list(model_names or zoo.keys())
    if set(names) - set(zoo):
        raise ValueError(f"Unknown models: {sorted(set(names) - set(zoo))}")
    root = Path(output_dir)
    for directory in ("cv", "student_test", "models"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    summary = {"protocol": {"cv": "GroupKFold grouped by train trip_id", "n_splits": n_splits,
                "seed": seed, "selection_metric": "mean observation_top1_accuracy",
                "training_sampling": "all positives and up to 12 near/random negatives per point; negative weights restore candidate counts",
                "student_test_role": "development-stage comparison only; not used for fitting, tuning or selection"},
                "feature_columns": list(FEATURE_COLUMNS), "models": {}}
    summary_path, csv_path = root / "model_comparison.json", root / "model_comparison.csv"
    fold_results = {name: [] for name in names}
    for name in names:
        saved_path = root / "cv" / f"{name}.json"
        if saved_path.exists():
            try:
                saved = json.loads(saved_path.read_text(encoding="utf-8"))
                fold_results[name] = saved.get("fold_metrics", [])
            except (OSError, json.JSONDecodeError):
                fold_results[name] = []
    for fold_no, (train_idx, valid_idx) in enumerate(folds, start=1):
        if all(len(fold_results[name]) >= fold_no for name in names):
            for name in names:
                current = fold_results[name][:fold_no]
                metric_keys = ["candidate_macro_f1", "candidate_log_loss", "observation_top1_accuracy",
                               "candidate_miss_rate", "route_continuity", "fit_seconds", "inference_seconds"]
                cv = {f"mean_{key}": float(np.mean([row[key] for row in current])) for key in metric_keys}
                cv.update({f"std_{key}": float(np.std([row[key] for row in current])) for key in metric_keys})
                summary["models"][name] = {"model": name,
                    "configuration": _json_safe(zoo[name].get_params(deep=False)),
                    "cv": cv, "cv_folds": current}
            print(f"[CV {fold_no}/{n_splits}] reusing saved fold results", flush=True)
            _save_json(summary_path, summary)
            _write_summary_csv(csv_path, summary)
            continue
        fit_start = time.perf_counter()
        x_train, y_train, weights, population_prior = _training_matrix(
            [train_paths[i] for i in train_idx], seed + fold_no)
        matrix_seconds = time.perf_counter() - fit_start
        print(f"[CV {fold_no}/{n_splits}] {len(y_train):,} sampled rows; "
              f"{len(train_idx)} train trips, {len(valid_idx)} validation trips", flush=True)
        for name in names:
            if len(fold_results[name]) >= fold_no:
                continue
            model = clone(zoo[name])
            fit_start = time.perf_counter()
            fitted, sample_prior = _fit_candidate_model(model, name, x_train, y_train,
                                                        weights, population_prior, seed + fold_no)
            fit_seconds = time.perf_counter() - fit_start
            score_start = time.perf_counter()
            scored = []
            for idx in valid_idx:
                data = load_trip_feature_set(train_paths[idx])
                scored.append(_score_trip(fitted, name, data, sample_prior, population_prior))
                del data
            inference_seconds = time.perf_counter() - score_start
            metrics = _merge_score_stats(scored)
            metrics.update({"fold": fold_no, "fit_seconds": fit_seconds,
                "training_matrix_seconds": matrix_seconds, "inference_seconds": inference_seconds,
                "inference_ms_per_observation": 1000*inference_seconds/max(1, metrics["observations_scored"])})
            fold_results[name].append(metrics)
            _save_json(root / "cv" / f"{name}.json", {"model": name, "folds_completed": fold_no,
                "fold_metrics": fold_results[name], "configuration": _json_safe(zoo[name].get_params(deep=False))})
            del fitted, model
        del x_train, y_train, weights
        for name in names:
            current = fold_results[name]
            metric_keys = ["candidate_macro_f1", "candidate_log_loss", "observation_top1_accuracy",
                           "candidate_miss_rate", "route_continuity", "fit_seconds", "inference_seconds"]
            cv = {f"mean_{key}": float(np.mean([row[key] for row in current])) for key in metric_keys}
            cv.update({f"std_{key}": float(np.std([row[key] for row in current])) for key in metric_keys})
            summary["models"][name] = {"model": name,
                "configuration": _json_safe(zoo[name].get_params(deep=False)),
                "cv": cv, "cv_folds": current}
        _save_json(summary_path, summary)
        _write_summary_csv(csv_path, summary)
    for name in names:
        print(f"[student comparison] fitting {name} on train trips only", flush=True)
        x_train, y_train, weights, population_prior = _training_matrix(train_paths, seed)
        model = clone(zoo[name])
        fit_start = time.perf_counter()
        fitted, sample_prior = _fit_candidate_model(model, name, x_train, y_train,
                                                    weights, population_prior, seed)
        fit_seconds = time.perf_counter() - fit_start
        del x_train, y_train, weights
        score_start = time.perf_counter()
        scored = []
        for path in student_paths:
            data = load_trip_feature_set(path)
            scored.append(_score_trip(fitted, name, data, sample_prior, population_prior))
            del data
        inference_seconds = time.perf_counter() - score_start
        student_metrics = _merge_score_stats(scored)
        student_metrics.update({"train_seconds": fit_seconds, "inference_seconds": inference_seconds,
            "inference_ms_per_observation": 1000*inference_seconds/max(1, student_metrics["observations_scored"])})
        model_path = root / "models" / f"{name}.joblib"
        joblib.dump({"model": fitted, "model_name": name, "feature_columns": list(FEATURE_COLUMNS),
            "sample_prior": sample_prior, "population_prior": population_prior,
            "candidate_radius_m": 250.0, "max_candidates": 128}, model_path)
        summary["models"][name]["student_test"] = student_metrics
        summary["models"][name]["model_artifact"] = str(model_path)
        _save_json(root / "student_test" / f"{name}.json", summary["models"][name])
        _save_json(summary_path, summary)
        _write_summary_csv(csv_path, summary)
        del fitted, model
    summary["selected_by_cv_top1"] = max(names,
        key=lambda n: summary["models"][n]["cv"]["mean_observation_top1_accuracy"])
    summary["selected_model_artifact"] = summary["models"][summary["selected_by_cv_top1"]]["model_artifact"]
    _save_json(summary_path, summary)
    _write_summary_csv(csv_path, summary)
    return summary


def tune_with_optuna(model_name: str, train_sets: Iterable[TripFeatureSet], output_dir: str | Path,
                     n_trials: int = 30, n_splits: int = 5, seed: int = 42) -> dict[str, Any]:
    """Optuna TPE search using only the 80 training trips and GroupKFold Top-1 accuracy."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for Bayesian optimisation; install the project dependencies first.") from exc
    train_sets = list(train_sets)
    data = flatten_sets(train_sets)
    folds = list(GroupKFold(n_splits=n_splits).split(data["X"], data["y"], data["groups"]))
    baseline = model_zoo(seed).get(model_name)
    if baseline is None:
        raise ValueError(f"Unsupported model for tuning: {model_name}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    def objective(trial):
        if model_name == "random_forest":
            params = {"n_estimators": trial.suggest_int("n_estimators", 100, 350),
                      "max_depth": trial.suggest_int("max_depth", 8, 48),
                      "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                      "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.7, 1.0]),
                      "class_weight": "balanced_subsample",
                      "random_state": seed, "n_jobs": -1}
            model = RandomForestClassifier(**params)
        elif model_name == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                learning_rate=trial.suggest_float("learning_rate", .025, .2, log=True),
                max_iter=trial.suggest_int("max_iter", 80, 250),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
                l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
                class_weight="balanced", random_state=seed)
        elif model_name == "logistic_regression":
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 100, log=True), max_iter=700,
                class_weight="balanced", random_state=seed))
        elif model_name == "gaussian_naive_bayes":
            model = make_pipeline(StandardScaler(), GaussianNB(priors=[0.5, 0.5],
                var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-7, log=True)))
        elif model_name == "knn":
            model = make_pipeline(StandardScaler(), KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", 5, 41, step=2),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                p=trial.suggest_int("p", 1, 2), n_jobs=-1))
        else:
            raise ValueError(f"No Optuna search space is defined for {model_name}.")
        fold_scores = []
        for tr, va in folds:
            fitted = clone(model)
            fitted.fit(data["X"][tr], data["y"][tr])
            p = _positive_probability(fitted, data["X"][va])
            valid_groups = set(data["groups"][va].tolist())
            valid_sets = [s for s in train_sets if s.trip_id in valid_groups]
            probs = {}
            for group in valid_groups:
                rows = np.flatnonzero(data["groups"][va] == group)
                probs[group] = p[rows]
            pred_map = _prediction_map(valid_sets, probs)
            fold_scores.append(_observation_metrics(valid_sets, pred_map)["observation_top1_accuracy"])
        trial.set_user_attr("fold_scores", fold_scores)
        return float(np.mean(fold_scores))

    study.optimize(objective, n_trials=n_trials)
    payload = {"model": model_name, "sampler": "Optuna TPESampler", "objective": "mean GroupKFold observation Top-1 accuracy",
               "train_groups_only": True, "n_splits": n_splits, "seed": seed,
               "best_value": study.best_value, "best_params": study.best_params,
               "trials": [{"number": t.number, "value": t.value, "params": t.params,
                           "fold_scores": t.user_attrs.get("fold_scores"), "state": t.state.name}
                          for t in study.trials]}
    _save_json(Path(output_dir) / "optuna_trials.json", payload)
    return payload
