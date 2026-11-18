"""Step 6: Bayesian hyperparameter optimisation (Optuna TPE = sequential model-based search).

Objective: mean observation-level Top-1 over trip-grouped folds of the TRAIN split (student-test
and evaluator data are never touched). Each fold reports an intermediate value so the median
pruner can stop clearly bad trials early. Every trial is stored in a SQLite study and exported
to CSV (the "optimisation search record").

Usage: python scripts/07_tune.py --model lightgbm --trials 40 --timeout-min 45 [--folds 3]
Outputs: results/tuning/optuna_<model>.db, trials_<model>.csv, best_<model>.json
Then compare tuned vs untuned with: python scripts/04_compare_models.py --models <model>_tuned
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.evaluate import point_keys, top1_accuracy
from pathsnap.features import FEATURES
from pathsnap.models import SPECS


def space(trial: optuna.Trial, model: str) -> dict:
    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            # lower bound ~0 so the untuned configuration (no L1/L2) is inside the space
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        }
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    raise ValueError(f"no search space for {model}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout-min", type=float, default=45)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()

    P = pd.read_parquet(config.CACHE_DIR / "dataset" / "train_pairs.parquet")
    Q = pd.read_parquet(config.CACHE_DIR / "dataset" / "train_points.parquet")
    codes = {t: i for i, t in enumerate(sorted(P["trip_id"].unique()))}
    P["_key"] = point_keys(P["trip_id"].map(codes).to_numpy(), P["obs"].to_numpy())
    X, y = P[FEATURES].to_numpy(), P["y"].to_numpy()
    rel, keys, trips = P["reliable"].to_numpy(), P["_key"].to_numpy(), P["trip_id"].to_numpy()
    n_rel_by_trip = Q[Q["reliable"]].groupby("trip_id").size()
    folds = list(GroupKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED).split(X, groups=trips))
    spec = SPECS[args.model]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial, args.model)
        scores = []
        t0 = time.perf_counter()
        for f, (tr, va) in enumerate(folds):
            tr = tr[rel[tr]]
            model = spec.build(params, config.SEED)
            model.fit(X[tr], y[tr])
            va_rel = va[rel[va]]
            prob = model.predict_proba(X[va_rel])[:, 1]
            n_rel = int(n_rel_by_trip.reindex(np.unique(trips[va])).fillna(0).sum())
            scores.append(top1_accuracy(keys[va_rel], y[va_rel], prob, n_rel))
            trial.report(float(np.mean(scores)), f)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_top1", scores)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean(scores))

    out = config.RESULTS_DIR / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"pathsnap_{args.model}", direction="maximize",
        storage=f"sqlite:///{(out / f'optuna_{args.model}.db').as_posix()}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.SEED, multivariate=True, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0))
    # trial 0 = the untuned configuration, so the search record starts from the baseline
    if len(study.trials) == 0:
        untuned = {k: v for k, v in spec.params.items() if k not in ("subsample_freq",)}
        if args.model == "lightgbm":
            untuned |= {"reg_lambda": 1e-8, "reg_alpha": 1e-8}  # LightGBM default is 0
        study.enqueue_trial(untuned)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout_min * 60)
    print(f"\n{len(study.trials)} trials in {(time.perf_counter() - t0) / 60:.1f} min; best top1 {study.best_value:.4f}")
    print("best params:", study.best_params)

    study.trials_dataframe().to_csv(out / f"trials_{args.model}.csv", index=False)
    best = dict(study.best_params)
    if args.model == "lightgbm":
        best["subsample_freq"] = 1
    (out / f"best_{args.model}.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
