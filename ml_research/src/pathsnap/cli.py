"""Command-line entry points for safe profiling and experiment workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .profiling import write_profile


def main() -> None:
    parser = argparse.ArgumentParser(prog="pathsnap")
    sub = parser.add_subparsers(dest="command", required=True)
    profile = sub.add_parser("profile", help="inventory and profile .track/.nodes/.arcs/.route files")
    profile.add_argument("--data", default="data/raw")
    profile.add_argument("--out", default="reports/dataset_profile.json")
    features = sub.add_parser("build-features", help="create or refresh train/student candidate feature caches")
    features.add_argument("--archive", required=True)
    features.add_argument("--selection", default="logs/candidate_selection_train.json")
    features.add_argument("--cache-dir", default="data/interim/features")
    train = sub.add_parser("train", help="build train/student features and compare grouped-CV models")
    train.add_argument("--archive", required=True, help="Student folder ZIP containing train and student_test trips")
    train.add_argument("--selection", default="logs/candidate_selection_train.json")
    train.add_argument("--cache-dir", default="data/interim/features")
    train.add_argument("--experiment-dir", default="experiments/model_comparison")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    baseline = sub.add_parser("baseline", help="score nearest-candidate baseline from existing feature caches")
    baseline.add_argument("--cache-dir", default="data/interim/features")
    baseline.add_argument("--experiment-dir", default="experiments/model_comparison")
    baseline.add_argument("--folds", type=int, default=5)
    hmm = sub.add_parser("hmm", help="compare grouped RF classifier-only and HMM/Viterbi routes")
    hmm.add_argument("--cache-dir", default="data/interim/features")
    hmm.add_argument("--experiment-dir", default="experiments/hmm")
    hmm.add_argument("--folds", type=int, default=5)
    hmm.add_argument("--seed", type=int, default=42)
    summarize_hmm = sub.add_parser("summarize-hmm", help="add candidate-level, calibration, and per-fold diagnostics to saved RF/HMM results")
    summarize_hmm.add_argument("--cache-dir", default="data/interim/features")
    summarize_hmm.add_argument("--experiment-dir", default="experiments/hmm")
    summarize_hmm.add_argument("--folds", type=int, default=5)
    xgb_smoke = sub.add_parser("xgb-smoke", help="small CUDA XGBoost fit to verify the GPU")
    xgb_smoke.add_argument("--out", default="experiments/xgboost_gpu/gpu_smoke.json")
    xgb = sub.add_parser("xgb-gpu", help="trip-grouped five-fold XGBoost run on CUDA")
    xgb.add_argument("--cache-dir", default="data/interim/features")
    xgb.add_argument("--experiment-dir", default="experiments/xgboost_gpu")
    xgb.add_argument("--folds", type=int, default=5)
    xgb.add_argument("--seed", type=int, default=42)
    diagnose = sub.add_parser("diagnose-trips", help="check reverse edges and map label alignment")
    diagnose.add_argument("--archive", required=True)
    diagnose.add_argument("--cache-dir", default="data/interim/features")
    diagnose.add_argument("--output-dir", default="reports/trip_diagnostics")
    diagnose.add_argument("--no-maps", action="store_true",
                          help="write numeric diagnostics without importing Folium")
    sub.add_parser("evaluate-final", help="score frozen predictions against evaluator labels once")
    args = parser.parse_args()
    if args.command == "profile":
        report = write_profile(args.data, args.out)
        print(json.dumps({"file_count": report.get("file_count", 0), "profile": args.out}, indent=2))
    elif args.command == "build-features":
        from .feature_store import build_features_for_split
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["selected"]
        train_paths = build_features_for_split(args.archive, "train", selection["radius_m"],
            selection["max_candidates"], args.cache_dir)
        student_paths = build_features_for_split(args.archive, "student_test", selection["radius_m"],
            selection["max_candidates"], args.cache_dir)
        print(json.dumps({"train_trip_caches": len(train_paths),
                          "student_test_trip_caches": len(student_paths),
                          "cache_dir": args.cache_dir}, indent=2))
    elif args.command == "train":
        from .feature_store import build_features_for_split
        from .training import compare_cached_models, evaluate_nearest_arc_baseline
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))["selected"]
        train_paths = build_features_for_split(args.archive, "train", selection["radius_m"],
            selection["max_candidates"], args.cache_dir)
        student_paths = build_features_for_split(args.archive, "student_test", selection["radius_m"],
            selection["max_candidates"], args.cache_dir)
        evaluate_nearest_arc_baseline(train_paths, student_paths, args.experiment_dir,
                                      n_splits=args.folds, seed=args.seed)
        result = compare_cached_models(train_paths, student_paths, args.experiment_dir,
                                       n_splits=args.folds, seed=args.seed)
        print(json.dumps({"selected_model": result["selected_by_cv_top1"],
                          "summary": str(Path(args.experiment_dir) / "model_comparison.json")}, indent=2))
    elif args.command == "baseline":
        from .training import evaluate_nearest_arc_baseline
        cache = Path(args.cache_dir)
        train_paths = sorted((cache / "train").glob("*.npz"))
        student_paths = sorted((cache / "student_test").glob("*.npz"))
        result = evaluate_nearest_arc_baseline(train_paths, student_paths, args.experiment_dir,
                                               n_splits=args.folds)
        print(json.dumps({"training_oof": result["training_oof"],
                          "student_test": result["student_test_development_comparison"]}, indent=2))
    elif args.command == "hmm":
        from .hmm_training import run_random_forest_hmm_experiment
        cache = Path(args.cache_dir)
        train_paths = sorted((cache / "train").glob("*.npz"))
        student_paths = sorted((cache / "student_test").glob("*.npz"))
        result = run_random_forest_hmm_experiment(train_paths, student_paths, args.experiment_dir,
                                                   n_splits=args.folds, seed=args.seed)
        print(json.dumps({"classifier_top1": result["classifier_only"]["all_training_oof"]["observation_top1_accuracy"],
                          "hmm_top1": result["classifier_plus_hmm"]["nested_training_oof"]["observation_top1_accuracy"],
                          "student_classifier_top1": result["student_test_development_comparison"]["classifier_only"]["observation_top1_accuracy"],
                          "student_hmm_top1": result["student_test_development_comparison"]["classifier_plus_hmm"]["observation_top1_accuracy"],
                          "report": str(Path(args.experiment_dir) / "classifier_vs_hmm.json")}, indent=2))
    elif args.command == "summarize-hmm":
        from .hmm_training import summarize_saved_rf_hmm
        result = summarize_saved_rf_hmm(args.cache_dir, args.experiment_dir, args.folds)
        print(json.dumps({"candidate_oof_logloss":
                          result["candidate_level_classifier_metrics"]["training_oof_pooled"]["candidate_log_loss"],
                          "probability_ece": result["probability_calibration"]["oof_equal_width_20_bin_ece"],
                          "hmm_hurt_trip_count": len(result["classifier_plus_hmm"]["trips_where_hmm_hurt"]),
                          "report": str(Path(args.experiment_dir) / "classifier_vs_hmm.json")}, indent=2))
    elif args.command == "xgb-smoke":
        from .xgboost_training import xgb_gpu_smoke_test
        result = xgb_gpu_smoke_test(args.out)
        print(json.dumps({"status": result["status"], "device": result["device"],
                          "xgboost_version": result["xgboost_version"],
                          "smoke_report": args.out}, indent=2))
    elif args.command == "xgb-gpu":
        from .xgboost_training import run_xgboost_gpu_cv
        cache = Path(args.cache_dir)
        train_paths = sorted((cache / "train").glob("*.npz"))
        student_paths = sorted((cache / "student_test").glob("*.npz"))
        result = run_xgboost_gpu_cv(train_paths, student_paths, args.experiment_dir,
                                    n_splits=args.folds, seed=args.seed)
        print(json.dumps({"cv_pooled_top1": result["cv"]["pooled"]["observation_top1_accuracy"],
                          "cv_mean_trip_top1": result["cv"]["mean_per_trip_top1"],
                          "student_test_top1": result["student_test_development_comparison"]["observation_top1_accuracy"],
                          "report": str(Path(args.experiment_dir) / "xgboost_cuda_cv.json")}, indent=2))
    elif args.command == "diagnose-trips":
        from .trip_diagnostics import diagnose_trips
        result = diagnose_trips(args.archive, args.cache_dir, args.output_dir,
                                create_maps=not args.no_maps)
        print(json.dumps({"overall_training": result["overall_training"],
                          "focus_trips": result["focus_trip_details"],
                          "maps": result["maps"],
                          "report": str(Path(args.output_dir) / "trip_label_diagnostics.json")}, indent=2))
    else:
        raise SystemExit("Final scoring requires frozen predictions and the evaluator-only label source.")


if __name__ == "__main__":
    main()
