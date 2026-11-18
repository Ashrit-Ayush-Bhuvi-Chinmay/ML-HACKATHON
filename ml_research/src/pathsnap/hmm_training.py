"""Group-aware classifier versus HMM route evaluation for cached PathSnap trips."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .decoding import decode_trip_hmm
from .feature_pipeline import FEATURE_COLUMNS
from .feature_store import load_trip_feature_set
from .training import (
    _fit_candidate_model,
    _json_safe,
    _observation_metric_summary,
    _observation_metrics,
    _prediction_map,
    _probability_for_model,
    _save_json,
    _training_matrix,
    model_zoo,
)


DECODER_GRID = [
    {"same_arc_score": -0.2, "connected_score": -0.7, "disconnected_score": disconnected,
     "distance_ratio_weight": ratio}
    for disconnected in (-8.0, -12.0, -16.0)
    for ratio in (0.0, 0.6)
]


def _candidate_probabilities(model, data, sample_prior: float,
                             population_prior: float, chunk_rows: int = 100_000) -> np.ndarray:
    output = np.empty(data.candidate_count, dtype=np.float32)
    for start in range(0, data.candidate_count, chunk_rows):
        stop = min(data.candidate_count, start + chunk_rows)
        output[start:stop] = _probability_for_model(
            model, "random_forest", data.features[start:stop], sample_prior, population_prior)
    return output


def _metric_for_edges(data, edge_predictions: np.ndarray) -> dict:
    mapping = {(data.trip_id, i): int(edge_id) for i, edge_id in enumerate(edge_predictions) if edge_id >= 0}
    return _observation_metrics([data], mapping)


def _metric_for_classifier(data, probabilities: np.ndarray) -> dict:
    prediction_map = _prediction_map([data], {data.trip_id: probabilities})
    return _observation_metrics([data], prediction_map)


def _summarize_with_folds(trip_rows: dict[str, dict], fold_by_trip: dict[str, int],
                          exclude_trip_ids: set[str] | None = None) -> dict:
    exclude_trip_ids = exclude_trip_ids or set()
    all_rows = [row for trip_id, row in trip_rows.items() if trip_id not in exclude_trip_ids]
    fold_summaries = []
    for fold in sorted(set(fold_by_trip.values())):
        rows = [trip_rows[trip_id] for trip_id, assigned in fold_by_trip.items()
                if assigned == fold and trip_id not in exclude_trip_ids and trip_id in trip_rows]
        if rows:
            fold_summaries.append({"fold": fold, **_observation_metric_summary(rows)})
    top1_values = [row["observation_top1_accuracy"] for row in fold_summaries]
    return {**_observation_metric_summary(all_rows),
            "fold_top1_mean": float(np.mean(top1_values)) if top1_values else 0.0,
            "fold_top1_std": float(np.std(top1_values)) if top1_values else 0.0,
            "folds": fold_summaries}


def run_random_forest_hmm_experiment(train_paths, student_paths, output_dir: str | Path,
                                     n_splits: int = 5, seed: int = 42) -> dict:
    """Produce OOF classifier/HMM metrics; choose decoder settings using other folds only."""
    train_paths, student_paths = list(map(Path, train_paths)), list(map(Path, student_paths))
    if not train_paths or not student_paths:
        raise ValueError("Both train and student feature caches are required.")
    root = Path(output_dir)
    probability_dir = root / "oof_probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)
    train_ids = [path.stem for path in train_paths]
    splitter = GroupKFold(n_splits=n_splits)
    folds = list(splitter.split(np.arange(len(train_paths)), groups=train_ids))
    fold_by_trip: dict[str, int] = {}
    classifier_rows: dict[str, dict] = {}
    fit_log: list[dict] = []
    zoo = model_zoo(seed)
    model_name = "random_forest"
    prior_log = root / "rf_oof_fit_log.json"
    if prior_log.exists():
        try:
            fit_log = json.loads(prior_log.read_text(encoding="utf-8")).get("folds", [])
        except (OSError, json.JSONDecodeError):
            fit_log = []

    for fold_no, (fit_idx, valid_idx) in enumerate(folds, start=1):
        validation_paths = [train_paths[i] for i in valid_idx]
        if all((probability_dir / f"{path.stem}.npy").exists() for path in validation_paths):
            for path in validation_paths:
                data = load_trip_feature_set(path)
                probabilities = np.load(probability_dir / f"{path.stem}.npy", allow_pickle=False)
                classifier_rows[path.stem] = _metric_for_classifier(data, probabilities)
                fold_by_trip[path.stem] = fold_no
                del data, probabilities
            print(f"[RF+HMM fold {fold_no}/{n_splits}] reusing saved held-out probabilities", flush=True)
            continue
        print(f"[RF+HMM fold {fold_no}/{n_splits}] preparing training rows", flush=True)
        matrix_start = time.perf_counter()
        x_train, y_train, weights, population_prior = _training_matrix(
            [train_paths[i] for i in fit_idx], seed + fold_no)
        sampled_rows = len(y_train)
        matrix_seconds = time.perf_counter() - matrix_start
        model = clone(zoo[model_name])
        fit_start = time.perf_counter()
        fitted, sample_prior = _fit_candidate_model(model, model_name, x_train, y_train,
                                                   weights, population_prior, seed + fold_no)
        fit_seconds = time.perf_counter() - fit_start
        del x_train, y_train, weights
        fold_start = time.perf_counter()
        for path in validation_paths:
            trip_id = path.stem
            data = load_trip_feature_set(path)
            probabilities = _candidate_probabilities(fitted, data, sample_prior, population_prior)
            np.save(probability_dir / f"{trip_id}.npy", probabilities)
            classifier_rows[trip_id] = _metric_for_classifier(data, probabilities)
            fold_by_trip[trip_id] = fold_no
            del probabilities, data
        inference_seconds = time.perf_counter() - fold_start
        fit_log = [row for row in fit_log if row.get("fold") != fold_no]
        fit_log.append({"fold": fold_no, "training_matrix_seconds": matrix_seconds,
                        "fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
                        "train_trip_count": len(fit_idx), "validation_trip_count": len(valid_idx),
                        "sampled_candidate_rows": int(sampled_rows)})
        _save_json(root / "rf_oof_fit_log.json", {"model": model_name, "folds_completed": fold_no,
                                                   "folds": fit_log})
        del fitted, model
        print(f"[RF+HMM fold {fold_no}/{n_splits}] saved held-out probabilities", flush=True)

    per_config: dict[str, dict[str, dict]] = {json.dumps(params, sort_keys=True): {} for params in DECODER_GRID}
    config_by_key = {json.dumps(params, sort_keys=True): params for params in DECODER_GRID}
    for path in train_paths:
        trip_id = path.stem
        data = load_trip_feature_set(path)
        probabilities = np.load(probability_dir / f"{trip_id}.npy", allow_pickle=False)
        for key, params in config_by_key.items():
            decoded = decode_trip_hmm(data, probabilities, **params)
            per_config[key][trip_id] = _metric_for_edges(data, decoded)
        del data, probabilities
        print(f"[HMM decode] trip={trip_id} settings={len(DECODER_GRID)}", flush=True)

    # For each held-out fold, select transition settings using only the other OOF folds.
    nested_hmm_rows: dict[str, dict] = {}
    nested_choices = []
    for held_out_fold in range(1, n_splits + 1):
        candidates = []
        for key, rows in per_config.items():
            tune_rows = [rows[trip_id] for trip_id, fold in fold_by_trip.items()
                         if fold != held_out_fold and trip_id in rows]
            tune_score = _observation_metric_summary(tune_rows)["observation_top1_accuracy"]
            candidates.append((tune_score, key))
        chosen_score, chosen_key = max(candidates, key=lambda item: (item[0], item[1]))
        nested_choices.append({"held_out_fold": held_out_fold, "selected_params": config_by_key[chosen_key],
                               "selection_top1_other_folds": chosen_score})
        for trip_id, fold in fold_by_trip.items():
            if fold == held_out_fold:
                nested_hmm_rows[trip_id] = per_config[chosen_key][trip_id]

    overall_scores = [( _observation_metric_summary(list(rows.values()))["observation_top1_accuracy"], key)
                      for key, rows in per_config.items()]
    selected_score, selected_key = max(overall_scores, key=lambda item: (item[0], item[1]))
    selected_params = config_by_key[selected_key]
    outlier_ids = [trip_id for trip_id, row in sorted(classifier_rows.items(),
        key=lambda item: (-item[1]["candidate_miss_rate"], item[0]))[:3]]

    classifier_all = _summarize_with_folds(classifier_rows, fold_by_trip)
    classifier_clean = _summarize_with_folds(classifier_rows, fold_by_trip, set(outlier_ids))
    hmm_all = _summarize_with_folds(nested_hmm_rows, fold_by_trip)
    hmm_clean = _summarize_with_folds(nested_hmm_rows, fold_by_trip, set(outlier_ids))

    # Refit on all training trips, freeze the selected decoder, then score student trips once.
    print("[student comparison] fitting RF on all training trips", flush=True)
    x_train, y_train, weights, population_prior = _training_matrix(train_paths, seed)
    final_model = clone(zoo[model_name])
    fit_start = time.perf_counter()
    final_model, sample_prior = _fit_candidate_model(final_model, model_name, x_train, y_train,
                                                     weights, population_prior, seed)
    final_fit_seconds = time.perf_counter() - fit_start
    del x_train, y_train, weights
    student_classifier_rows, student_hmm_rows = {}, {}
    student_inference_start = time.perf_counter()
    for path in student_paths:
        data = load_trip_feature_set(path)
        probabilities = _candidate_probabilities(final_model, data, sample_prior, population_prior)
        student_classifier_rows[data.trip_id] = _metric_for_classifier(data, probabilities)
        decoded = decode_trip_hmm(data, probabilities, **selected_params)
        student_hmm_rows[data.trip_id] = _metric_for_edges(data, decoded)
        del data, probabilities
    student_inference_seconds = time.perf_counter() - student_inference_start

    import joblib
    artifact = root / "random_forest_hmm.joblib"
    joblib.dump({"model": final_model, "model_name": model_name,
        "feature_columns": list(FEATURE_COLUMNS), "sample_prior": sample_prior,
        "population_prior": population_prior, "candidate_radius_m": 250.0,
        "max_candidates": 128, "decoder_params": selected_params}, artifact)

    result = {"protocol": {"model": "random_forest", "cv": "GroupKFold by train trip_id",
                "n_splits": n_splits, "seed": seed, "decoder_selection": "for each held-out fold, select settings on the other four OOF folds",
                "probabilities": "sample-weighted candidate probabilities, prevalence corrected, normalized within point by the HMM",
                "student_test_role": "development comparison only; not used to fit or tune"},
        "classifier_only": {"all_training_oof": classifier_all,
            "excluding_three_highest_candidate_miss_trips": classifier_clean,
            "excluded_trip_ids": outlier_ids},
        "classifier_plus_hmm": {"nested_training_oof": hmm_all,
            "excluding_three_highest_candidate_miss_trips": hmm_clean,
            "excluded_trip_ids": outlier_ids},
        "student_test_development_comparison": {
            "classifier_only": _observation_metric_summary(list(student_classifier_rows.values())),
            "classifier_plus_hmm": _observation_metric_summary(list(student_hmm_rows.values())),
            "train_fit_seconds": final_fit_seconds, "inference_seconds": student_inference_seconds},
        "selected_decoder_params_from_all_training_oof": selected_params,
        "decoder_grid_results": [{"params": config_by_key[key],
            "training_oof_top1": score} for score, key in sorted(overall_scores, reverse=True)],
        "nested_decoder_choices": nested_choices, "rf_fold_fit_log": fit_log,
        "model_artifact": str(artifact)}
    _save_json(root / "classifier_vs_hmm.json", result)
    with (root / "hmm_trip_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "trip_id", "classifier_top1", "hmm_top1", "candidate_miss_rate"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trip_id in classifier_rows:
            writer.writerow({"split": "train_oof", "trip_id": trip_id,
                "classifier_top1": classifier_rows[trip_id]["observation_top1_accuracy"],
                "hmm_top1": nested_hmm_rows[trip_id]["observation_top1_accuracy"],
                "candidate_miss_rate": classifier_rows[trip_id]["candidate_miss_rate"]})
        for trip_id in student_classifier_rows:
            writer.writerow({"split": "student_test", "trip_id": trip_id,
                "classifier_top1": student_classifier_rows[trip_id]["observation_top1_accuracy"],
                "hmm_top1": student_hmm_rows[trip_id]["observation_top1_accuracy"],
                "candidate_miss_rate": student_classifier_rows[trip_id]["candidate_miss_rate"]})
    return result


def summarize_saved_rf_hmm(cache_dir: str | Path, output_dir: str | Path,
                           n_splits: int = 5) -> dict:
    """Add candidate loss/F1, calibration diagnostics, pooled trip means, and fold files."""
    import joblib
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    train_paths = sorted((cache_dir / "train").glob("*.npz"))
    student_paths = sorted((cache_dir / "student_test").glob("*.npz"))
    report_path = output_dir / "classifier_vs_hmm.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trip_csv = output_dir / "hmm_trip_metrics.csv"
    with trip_csv.open(newline="", encoding="utf-8") as stream:
        trip_rows = list(csv.DictReader(stream))
    trip_by_key = {(row["split"], row["trip_id"]): row for row in trip_rows}
    ids = [p.stem for p in train_paths]
    fold_by_trip = {}
    for fold_no, (_, valid_idx) in enumerate(
            GroupKFold(n_splits=n_splits).split(np.arange(len(ids)), groups=ids), start=1):
        for idx in valid_idx:
            fold_by_trip[ids[idx]] = fold_no

    candidate_rows = []
    probability_bin_count = np.zeros(20, dtype=np.int64)
    probability_bin_sum = np.zeros(20, dtype=np.float64)
    probability_bin_positive = np.zeros(20, dtype=np.float64)
    for path in train_paths:
        data = load_trip_feature_set(path)
        probabilities = np.load(output_dir / "oof_probabilities" / f"{path.stem}.npy", allow_pickle=False)
        labels = data.labels.astype(np.int8)
        predicted = probabilities >= 0.5
        positive = labels == 1
        tp = int(np.count_nonzero(predicted & positive))
        fp = int(np.count_nonzero(predicted & ~positive))
        fn = int(np.count_nonzero(~predicted & positive))
        tn = int(np.count_nonzero(~predicted & ~positive))
        loss_sum = float(-np.sum(labels*np.log(np.clip(probabilities, 1e-7, 1-1e-7)) +
                                 (1-labels)*np.log1p(-np.clip(probabilities, 1e-7, 1-1e-7)),
                                 dtype=np.float64))
        brier_sum = float(np.sum((probabilities - labels)**2, dtype=np.float64))
        row = {"trip_id": data.trip_id, "fold": fold_by_trip[data.trip_id],
            "candidate_rows": data.candidate_count, "positive_rows": int(np.count_nonzero(positive)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "loss_sum": loss_sum,
            "brier_sum": brier_sum, "probability_sum": float(probabilities.sum(dtype=np.float64))}
        candidate_rows.append(row)
        bins = np.minimum((probabilities * len(probability_bin_count)).astype(np.int64),
                          len(probability_bin_count)-1)
        probability_bin_count += np.bincount(bins, minlength=len(probability_bin_count))
        probability_bin_sum += np.bincount(bins, weights=probabilities,
                                            minlength=len(probability_bin_count))
        probability_bin_positive += np.bincount(bins, weights=labels,
                                                minlength=len(probability_bin_count))
        del data, probabilities

    def aggregate(rows):
        keys = ("candidate_rows", "positive_rows", "tp", "fp", "fn", "tn", "loss_sum",
                "brier_sum", "probability_sum")
        sums = {key: sum(row[key] for row in rows) for key in keys}
        tp, fp, fn, tn = sums["tp"], sums["fp"], sums["fn"], sums["tn"]
        f1_pos = 2*tp / max(1, 2*tp+fp+fn)
        f1_neg = 2*tn / max(1, 2*tn+fp+fn)
        return {"candidate_rows": sums["candidate_rows"], "positive_rows": sums["positive_rows"],
            "candidate_accuracy": (tp+tn)/max(1, sums["candidate_rows"]),
            "candidate_macro_f1": (f1_pos+f1_neg)/2,
            "candidate_precision": tp/max(1, tp+fp), "candidate_recall": tp/max(1, tp+fn),
            "candidate_log_loss": sums["loss_sum"]/max(1, sums["candidate_rows"]),
            "brier_score": sums["brier_sum"]/max(1, sums["candidate_rows"]),
            "mean_predicted_positive_probability": sums["probability_sum"]/max(1, sums["candidate_rows"]),
            "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp]}

    oof_candidate = aggregate(candidate_rows)
    actual_positive_rate = oof_candidate["positive_rows"] / max(1, oof_candidate["candidate_rows"])
    ece = 0.0
    reliability = []
    total_probability_rows = int(probability_bin_count.sum())
    for idx, count in enumerate(probability_bin_count):
        if not count:
            continue
        mean_p = probability_bin_sum[idx] / count
        observed = probability_bin_positive[idx] / count
        ece += count / max(1, total_probability_rows) * abs(mean_p-observed)
        reliability.append({"bin_low": idx/len(probability_bin_count),
            "bin_high": (idx+1)/len(probability_bin_count), "count": int(count),
            "mean_probability": float(mean_p), "observed_positive_rate": float(observed)})

    trip_map = {row["trip_id"]: row for row in trip_rows if row["split"] == "train_oof"}
    outliers = {"00000066", "00000002", "00000054"}
    for section, summary_key, csv_key in (
        ("classifier_only", "all_training_oof", "classifier_top1"),
        ("classifier_plus_hmm", "nested_training_oof", "hmm_top1")):
        records = [trip_map[row["trip_id"]] for row in candidate_rows]
        tops = [float(record[csv_key]) for record in records]
        report[section][summary_key]["mean_per_trip_top1"] = float(np.mean(tops))
        report[section][summary_key]["per_trip_top1"] = [
            {"trip_id": r["trip_id"], "top1": float(r[csv_key])} for r in records]
        report[section][summary_key]["fold_mean_per_trip_top1"] = {
            str(fold): float(np.mean([float(trip_map[trip_id][csv_key])
                for trip_id, assigned in fold_by_trip.items() if assigned == fold]))
            for fold in range(1, n_splits+1)}
        report[section][summary_key]["mean_per_trip_top1"] = float(np.mean(tops))
        excluded_records = [r for r in records if r["trip_id"] not in outliers]
        report[section]["excluding_three_highest_candidate_miss_trips"]["mean_per_trip_top1"] = float(
            np.mean([float(r[csv_key]) for r in excluded_records]))
        # Make the per-trip classifier and HMM comparison easy to audit.
        report[section]["excluding_three_highest_candidate_miss_trips"]["excluded_trip_ids"] = sorted(outliers)

    classifier_hurt, student_hurt = [], []
    for row in trip_rows:
        delta = float(row["hmm_top1"]) - float(row["classifier_top1"])
        if delta < 0:
            (classifier_hurt if row["split"] == "train_oof" else student_hurt).append(
                {"trip_id": row["trip_id"], "classifier_top1": float(row["classifier_top1"]),
                 "hmm_top1": float(row["hmm_top1"]), "delta": delta})
    report["classifier_plus_hmm"]["trips_where_hmm_hurt"] = sorted(
        classifier_hurt, key=lambda row: row["delta"])
    report["student_test_development_comparison"]["trips_where_hmm_hurt"] = sorted(
        student_hurt, key=lambda row: row["delta"])
    report["candidate_level_classifier_metrics"] = {
        "training_oof_pooled": oof_candidate,
        "training_oof_without_trips_66_02_54": aggregate(
            [row for row in candidate_rows if row["trip_id"] not in outliers]),
        "per_fold": {}, "per_fold_without_trips_66_02_54": {},
        "per_trip": []}
    decode_seconds_by_fold = {}
    for fold in range(1, n_splits+1):
        fold_rows = [row for row in candidate_rows if row["fold"] == fold]
        metrics = aggregate(fold_rows)
        fit_record = next(row for row in report["rf_fold_fit_log"] if row["fold"] == fold)
        trip_fold = [trip_map[row["trip_id"]] for row in fold_rows]
        metrics.update({"observation_top1_pooled":
            report["classifier_only"]["all_training_oof"]["folds"][fold-1]["observation_top1_accuracy"],
            "mean_per_trip_top1": float(np.mean([float(r["classifier_top1"]) for r in trip_fold])),
            "fit_seconds": fit_record["fit_seconds"],
            "inference_seconds": fit_record["inference_seconds"]})
        report["candidate_level_classifier_metrics"]["per_fold"][str(fold)] = metrics
        clean_fold_rows = [row for row in fold_rows if row["trip_id"] not in outliers]
        report["candidate_level_classifier_metrics"]["per_fold_without_trips_66_02_54"][str(fold)] = (
            aggregate(clean_fold_rows) if clean_fold_rows else {})
        fold_file = output_dir / "cv" / f"fold_{fold}.json"
        if fold_file.exists():
            try:
                decode_seconds_by_fold[fold] = json.loads(
                    fold_file.read_text(encoding="utf-8")).get("hmm_decode_seconds", 0.0)
            except (OSError, json.JSONDecodeError):
                decode_seconds_by_fold[fold] = 0.0
        else:
            decoder_params = report["nested_decoder_choices"][fold-1]["selected_params"]
            fold_start = time.perf_counter()
            for trip_id, assigned in fold_by_trip.items():
                if assigned != fold:
                    continue
                data = load_trip_feature_set(cache_dir / "train" / f"{trip_id}.npz")
                p = np.load(output_dir / "oof_probabilities" / f"{trip_id}.npy", allow_pickle=False)
                decode_trip_hmm(data, p, **decoder_params)
            decode_seconds_by_fold[fold] = time.perf_counter() - fold_start
        fold_json = {"fold": fold, "classifier_top1": report["classifier_only"]["all_training_oof"]["folds"][fold-1],
            "classifier_plus_hmm": report["classifier_plus_hmm"]["nested_training_oof"]["folds"][fold-1],
            "candidate_classifier_metrics": metrics,
            "fit_log": fit_record, "hmm_decode_seconds": decode_seconds_by_fold[fold]}
        cv_dir = output_dir / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)
        _save_json(cv_dir / f"fold_{fold}.json", fold_json)
    report["probability_calibration"] = {
        "method": "Inverse-negative-sampling weights were used in RandomForest fitting; no separate held-out calibrator was applied.",
        "candidate_population_positive_rate": actual_positive_rate,
        "oof_mean_predicted_positive_probability": oof_candidate["mean_predicted_positive_probability"],
        "oof_brier_score": oof_candidate["brier_score"],
        "oof_candidate_log_loss": oof_candidate["candidate_log_loss"],
        "oof_equal_width_20_bin_ece": float(ece), "reliability_bins": reliability}

    # Student candidate metrics are computed with the frozen all-training RF artifact.
    artifact_info = joblib.load(report["model_artifact"])
    student_candidate_rows = []
    student_classifier_decode_seconds = student_hmm_decode_seconds = 0.0
    student_hurt = []
    selected_params = report["selected_decoder_params_from_all_training_oof"]
    for path in student_paths:
        data = load_trip_feature_set(path)
        model = artifact_info["model"]
        from .training import _probability_for_model
        p = np.empty(data.candidate_count, dtype=np.float32)
        prediction_start = time.perf_counter()
        for start in range(0, data.candidate_count, 100_000):
            stop = min(data.candidate_count, start+100_000)
            p[start:stop] = _probability_for_model(model, "random_forest",
                data.features[start:stop], artifact_info["sample_prior"], artifact_info["population_prior"])
        student_classifier_decode_seconds += time.perf_counter() - prediction_start
        prediction_start = time.perf_counter()
        decoded = decode_trip_hmm(data, p, **selected_params)
        student_hmm_decode_seconds += time.perf_counter() - prediction_start
        labels = data.labels
        pred, pos = p >= .5, labels == 1
        tp, fp = int(np.count_nonzero(pred & pos)), int(np.count_nonzero(pred & ~pos))
        fn, tn = int(np.count_nonzero(~pred & pos)), int(np.count_nonzero(~pred & ~pos))
        clipped = np.clip(p,1e-7,1-1e-7)
        student_candidate_rows.append({"candidate_rows": data.candidate_count,
            "positive_rows": int(np.count_nonzero(pos)), "tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "loss_sum":float(-np.sum(labels*np.log(clipped)+(1-labels)*np.log1p(-clipped))),
            "brier_sum":float(np.sum((p-labels)**2)),
            "probability_sum":float(p.sum(dtype=np.float64))})
        del data, p
    report["candidate_level_classifier_metrics"]["student_test_development_comparison"] = aggregate(
        student_candidate_rows)
    for row in candidate_rows:
        single = aggregate([row])
        single.update({"trip_id": row["trip_id"], "fold": row["fold"]})
        report["candidate_level_classifier_metrics"]["per_trip"].append(single)
    student_trip_top1 = [float(row["classifier_top1"]) for row in trip_rows if row["split"] == "student_test"]
    student_hmm_trip_top1 = [float(row["hmm_top1"]) for row in trip_rows if row["split"] == "student_test"]
    report["student_test_development_comparison"]["classifier_only"]["mean_per_trip_top1"] = float(
        np.mean(student_trip_top1))
    report["student_test_development_comparison"]["classifier_plus_hmm"]["mean_per_trip_top1"] = float(
        np.mean(student_hmm_trip_top1))
    report["student_test_development_comparison"]["classifier_inference_seconds_rechecked"] = student_classifier_decode_seconds
    report["student_test_development_comparison"]["hmm_decode_seconds_rechecked"] = student_hmm_decode_seconds
    report["classifier_plus_hmm"]["nested_training_oof"]["hmm_decode_seconds_total_rechecked"] = float(
        sum(decode_seconds_by_fold.values()))
    _save_json(report_path, report)
    return report
