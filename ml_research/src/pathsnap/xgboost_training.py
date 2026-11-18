"""CUDA XGBoost smoke test and trip-grouped model comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .feature_pipeline import FEATURE_COLUMNS
from .feature_store import load_trip_feature_set
from .training import _merge_score_stats, _save_json, _score_trip, _training_matrix


def _xgb_classifier(seed: int, device: str = "cuda"):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Install xgboost in the active environment before using the GPU commands.") from exc
    return XGBClassifier(
        n_estimators=180, max_depth=7, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device=device, random_state=seed,
        n_jobs=8, verbosity=0,
    )


def xgb_gpu_smoke_test(output_path: str | Path | None = None) -> dict:
    """Run a small fit and verify the serialized booster is configured for CUDA."""
    try:
        import xgboost
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed in this Python environment.") from exc
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1024, len(FEATURE_COLUMNS))).astype(np.float32)
    y = ((x[:, 0] - 0.7 * x[:, 3] + 0.2 * x[:, 10]) > 0).astype(np.int8)
    model = _xgb_classifier(seed=42, device="cuda")
    started = time.perf_counter()
    model.fit(x, y, sample_weight=np.ones_like(y, dtype=np.float32), verbose=False)
    probabilities = model.predict_proba(x[:32])
    config_text = model.get_booster().save_config()
    build = xgboost.build_info() if hasattr(xgboost, "build_info") else {}
    configured_cuda = '"cuda"' in config_text or '"cuda:0"' in config_text
    build_cuda = bool(build.get("USE_CUDA", False))
    if not configured_cuda or (build and not build_cuda):
        raise RuntimeError(f"XGBoost did not confirm CUDA use. device_config={configured_cuda}, build={build}")
    result = {"status": "passed", "xgboost_version": xgboost.__version__,
        "device": "cuda:0", "cuda_build": build_cuda if build else "not reported by this XGBoost version",
        "rows_fit": len(y), "prediction_shape": list(probabilities.shape),
        "fit_seconds": time.perf_counter() - started, "parameters": model.get_params()}
    if output_path is not None:
        _save_json(Path(output_path), result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _pooled_fold_metrics(folds: list[dict]) -> dict:
    tn = fp = fn = tp = candidate_rows = observation_count = correct = misses = covered = 0
    continuity_pairs = connected_pairs = 0
    loss_total = 0.0
    fit_seconds = inference_seconds = 0.0
    for fold in folds:
        f_tn, f_fp, f_fn, f_tp = fold["confusion_matrix_tn_fp_fn_tp"]
        tn += f_tn
        fp += f_fp
        fn += f_fn
        tp += f_tp
        candidate_rows += fold["candidate_rows_scored"]
        loss_total += fold["candidate_log_loss"] * fold["candidate_rows_scored"]
        observation_count += fold["observations_scored"]
        correct += fold["observation_top1_accuracy"] * fold["observations_scored"]
        misses += fold["candidate_miss_rate"] * fold["observations_scored"]
        covered += fold["prediction_coverage"] * fold["observations_scored"]
        continuity_pairs += fold["route_continuity_pairs_scored"]
        connected_pairs += fold["route_continuity"] * fold["route_continuity_pairs_scored"]
        fit_seconds += fold["fit_seconds"]
        inference_seconds += fold["inference_seconds"]
    f1_pos = 2 * tp / max(1, 2 * tp + fp + fn)
    f1_neg = 2 * tn / max(1, 2 * tn + fp + fn)
    return {"candidate_accuracy": (tp + tn) / max(1, candidate_rows),
        "candidate_macro_f1": (f1_pos + f1_neg) / 2,
        "candidate_log_loss": loss_total / max(1, candidate_rows),
        "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
        "observation_top1_accuracy": correct / max(1, observation_count),
        "candidate_miss_rate": misses / max(1, observation_count),
        "prediction_coverage": covered / max(1, observation_count),
        "route_continuity": connected_pairs / max(1, continuity_pairs),
        "route_continuity_pairs_scored": continuity_pairs,
        "observations_scored": observation_count, "candidate_rows_scored": candidate_rows,
        "train_seconds_total": fit_seconds, "inference_seconds_total": inference_seconds}


def run_xgboost_gpu_cv(train_paths, student_paths, output_dir: str | Path,
                       n_splits: int = 5, seed: int = 42) -> dict:
    """Run trip-grouped CV on CUDA, scoring every validation candidate."""
    import xgboost
    train_paths, student_paths = list(map(Path, train_paths)), list(map(Path, student_paths))
    if not train_paths or not student_paths:
        raise ValueError("Both train and student feature caches are required.")
    root = Path(output_dir)
    (root / "cv").mkdir(parents=True, exist_ok=True)
    ids = [path.stem for path in train_paths]
    splits = list(GroupKFold(n_splits=n_splits).split(np.arange(len(ids)), groups=ids))
    fold_results, trip_results = [], []
    for fold_no, (fit_idx, valid_idx) in enumerate(splits, start=1):
        print(f"[XGBoost CUDA fold {fold_no}/{n_splits}] building train rows", flush=True)
        x, y, weights, _ = _training_matrix([train_paths[i] for i in fit_idx], seed + fold_no)
        model = _xgb_classifier(seed + fold_no)
        fit_start = time.perf_counter()
        model.fit(x, y, sample_weight=weights, verbose=False)
        fit_seconds = time.perf_counter() - fit_start
        del x, y, weights
        inference_start = time.perf_counter()
        validation_stats = []
        fold_trip_results = []
        for idx in valid_idx:
            data = load_trip_feature_set(train_paths[idx])
            stats = _score_trip(model, "xgboost", data, 0.0, 0.0)
            stats["trip_id"] = data.trip_id
            stats["split"] = "train_oof"
            stats["fold"] = fold_no
            validation_stats.append(stats)
            fold_trip_results.append(stats)
            del data
        inference_seconds = time.perf_counter() - inference_start
        metrics = _merge_score_stats(validation_stats)
        metrics.update({"fold": fold_no, "fit_seconds": fit_seconds,
                        "inference_seconds": inference_seconds,
                        "inference_ms_per_observation": 1000*inference_seconds /
                            max(1, metrics["observations_scored"])})
        fold_results.append(metrics)
        trip_results.extend(fold_trip_results)
        _save_json(root / "cv" / f"fold_{fold_no}.json", {"fold": fold_no,
            "model": "xgboost", "device": "cuda:0", "metrics": metrics,
            "trip_metrics": fold_trip_results})
        print(f"[XGBoost CUDA fold {fold_no}/{n_splits}] saved metrics; "
              f"top1={metrics['observation_top1_accuracy']:.4f}", flush=True)

    pooled = _pooled_fold_metrics(fold_results)
    trip_ids = [row["trip_id"] for row in trip_results]
    outliers = {"00000066", "00000002", "00000054"}
    all_trip_rows = [row for row in trip_results if row["trip_id"] not in outliers]
    pooled_trip_without_outliers = _merge_score_stats(all_trip_rows)
    per_trip_top1 = [row["point_correct"] / max(1, row["point_total"]) for row in trip_results]
    cv = {"pooled": pooled, "mean_fold": {
        key: float(np.mean([fold[key] for fold in fold_results]))
        for key in ("candidate_macro_f1", "candidate_log_loss", "observation_top1_accuracy",
                    "candidate_miss_rate", "route_continuity", "fit_seconds", "inference_seconds")},
        "fold_std": {key: float(np.std([fold[key] for fold in fold_results]))
        for key in ("candidate_macro_f1", "candidate_log_loss", "observation_top1_accuracy",
                    "candidate_miss_rate", "route_continuity", "fit_seconds", "inference_seconds")},
        "mean_per_trip_top1": float(np.mean(per_trip_top1)),
        "without_trips_66_02_54": pooled_trip_without_outliers}

    print("[XGBoost CUDA] fitting final model on training trips only", flush=True)
    x, y, weights, _ = _training_matrix(train_paths, seed)
    final_model = _xgb_classifier(seed)
    fit_start = time.perf_counter()
    final_model.fit(x, y, sample_weight=weights, verbose=False)
    final_fit_seconds = time.perf_counter() - fit_start
    del x, y, weights
    student_stats = []
    student_trip_results = []
    inference_start = time.perf_counter()
    for path in student_paths:
        data = load_trip_feature_set(path)
        stats = _score_trip(final_model, "xgboost", data, 0.0, 0.0)
        stats.update({"trip_id": data.trip_id, "split": "student_test"})
        student_stats.append(stats)
        student_trip_results.append(stats)
        del data
    student_inference_seconds = time.perf_counter() - inference_start
    student = _merge_score_stats(student_stats)
    student.update({"train_seconds": final_fit_seconds,
                    "inference_seconds": student_inference_seconds,
                    "mean_per_trip_top1": float(np.mean([
                        row["point_correct"] / max(1, row["point_total"]) for row in student_stats]))})
    model_path = root / "xgboost_cuda.ubj"
    final_model.save_model(model_path)
    result = {"protocol": {"model": "XGBoost", "device": "cuda:0",
        "tree_method": "hist", "cv": "GroupKFold by train trip_id", "n_splits": n_splits,
        "seed": seed, "student_test_role": "comparison only; not used for fitting or tuning",
        "feature_columns": list(FEATURE_COLUMNS),
        "training_sampling": "all positives and up to 12 near/random negatives per point with inverse-sampling weights"},
        "cv": cv, "student_test_development_comparison": student,
        "student_test_trips": student_trip_results, "folds": fold_results,
        "train_oof_trips": trip_results, "model_artifact": str(model_path),
        "xgboost_version": xgboost.__version__}
    _save_json(root / "xgboost_cuda_cv.json", result)
    return result
