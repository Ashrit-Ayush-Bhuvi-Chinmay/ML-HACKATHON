"""Step 5 driver: compare models with trip-grouped 5-fold CV on the train split, then refit on
all train trips and evaluate on the student-test split.

Training uses reliably labelled points only; predictions are made for every candidate (so the
classifier-only route can be built for every point), and metrics use reliable points only.

Outputs
    results/models/model_comparison.csv      one row per model (latest run of each)
    results/models/runs/<model>_<stamp>.json config, seed, per-fold metrics, timings, histories
    cache/preds/<model>_{oof,test}.parquet   predictions for the HMM stage
Usage: python scripts/04_compare_models.py [--models lightgbm logreg ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.evaluate import pair_metrics, point_keys, route_continuity, top1_accuracy, best_per_point
from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.models import SPECS, DistanceGaussian

N_FOLDS = 5


def load(split):
    P = pd.read_parquet(config.CACHE_DIR / "dataset" / f"{split}_pairs.parquet")
    Q = pd.read_parquet(config.CACHE_DIR / "dataset" / f"{split}_points.parquet")
    return P, Q


def subsample_rows(P: pd.DataFrame, frac: float, rng) -> np.ndarray:
    """Row indices of a random subset of points (all candidates of each chosen point)."""
    if frac >= 1.0:
        return np.arange(len(P))
    keys = P["_key"].to_numpy()
    uniq = np.unique(keys)
    chosen = rng.choice(uniq, size=max(1, int(frac * len(uniq))), replace=False)
    return np.flatnonzero(np.isin(keys, chosen))


def fit_predict(spec, P_tr: pd.DataFrame, P_pred: pd.DataFrame, seed: int, rng):
    """Fit on reliable rows of P_tr, return (probabilities for P_pred, model, fit seconds, predict seconds)."""
    tr = P_tr[P_tr["reliable"]]
    if spec.build is None:  # distance-only baseline
        t0 = time.perf_counter()
        model = DistanceGaussian().fit(tr.loc[tr["y"] == 1, "dist"].to_numpy())
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        prob = model.predict_proba_grouped(P_pred["dist"].to_numpy(), P_pred["_key"].to_numpy())
        return prob, model, fit_s, time.perf_counter() - t0
    feats = spec.features or FEATURES
    idx = subsample_rows(tr, spec.train_point_frac, rng)
    X = tr[feats].to_numpy()[idx]
    y = tr["y"].to_numpy()[idx]
    model = spec.build(spec.params, seed)
    t0 = time.perf_counter()
    if hasattr(model, "fit_grouped"):  # needs each point's candidates together (torch models)
        model.fit_grouped(X, y, tr["_key"].to_numpy()[idx])
    else:
        model.fit(X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if hasattr(model, "predict_proba_grouped"):
        prob = model.predict_proba_grouped(P_pred[feats].to_numpy(), P_pred["_key"].to_numpy())
    else:
        prob = model.predict_proba(P_pred[feats].to_numpy())[:, 1]
    return prob, model, fit_s, time.perf_counter() - t0


def save_model(spec, model, row: dict) -> None:
    """Save the exact model whose student-test metrics are reported, and register it, so the
    API/UI can run (and show the results of) every model we tried, not only the winner."""
    art = config.REPO_ROOT / "artifacts" / "models"
    art.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "net_"):
        model.net_.cpu()  # GPU-trained torch weights saved on CPU: loadable on a laptop without CUDA
    bundle = {"model": model, "model_name": spec.name, "family": spec.family, "params": spec.params,
              "features": spec.features or FEATURES,
              "kind": "distance_gaussian" if spec.build is None else "classifier",
              "candidate_radius": config.CAND_RADIUS, "candidate_k": config.CAND_K,
              "metrics": row, "seed": config.SEED}
    path = art / f"{spec.name}.joblib"
    joblib.dump(bundle, path, compress=3)
    size_mb = path.stat().st_size / 1e6
    reg_path = art / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    reg[spec.name] = {k: row[k] for k in row if k != "config"} | {"file": path.name, "size_mb": round(size_mb, 2),
                                                                   "features": len(bundle["features"])}
    reg_path.write_text(json.dumps(reg, indent=2, default=str))
    print(f"  saved {path.name} ({size_mb:.1f} MB)", flush=True)


def metrics(P: pd.DataFrame, Q: pd.DataFrame, prob: np.ndarray) -> dict:
    rel = P["reliable"].to_numpy()
    m = pair_metrics(P["y"].to_numpy()[rel], prob[rel])
    trips = P["trip_id"].unique()
    n_rel = int(Q.loc[Q["trip_id"].isin(trips), "reliable"].sum())
    m["top1"] = top1_accuracy(P["_key"].to_numpy()[rel], P["y"].to_numpy()[rel], prob[rel], n_rel)
    return m


def continuity(P: pd.DataFrame, prob: np.ndarray, split: str) -> float:
    vals = []
    for trip_id, g in P.groupby("trip_id", sort=False):
        idx = g.index.to_numpy()
        _, best = best_per_point(P["_key"].to_numpy()[idx], prob[idx])
        arcs = load_trip(config.DATA_ROOT / split / trip_id).arcs
        vals.append(route_continuity(P["arc_id"].to_numpy()[idx][best], arcs))
    return float(np.mean(vals))


def get_spec(name: str):
    """SPECS entry, or '<model>_tuned' = that model with results/tuning/best_<model>.json params."""
    if name.endswith("_tuned"):
        base = SPECS[name[: -len("_tuned")]]
        best = json.loads((config.RESULTS_DIR / "tuning" / f"best_{base.name}.json").read_text())
        return dataclasses.replace(base, name=name, family=base.family + " (Bayesian-tuned)",
                                   params={**base.params, **best})
    return SPECS[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    args = ap.parse_args()

    P_tr, Q_tr = load("train")
    P_te, Q_te = load("student_test")
    codes = {t: i for i, t in enumerate(sorted(set(P_tr["trip_id"]) | set(P_te["trip_id"])))}
    for P in (P_tr, P_te):
        P["_key"] = point_keys(P["trip_id"].map(codes).to_numpy(), P["obs"].to_numpy())
    groups = P_tr["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED).split(P_tr, groups=groups))
    print(f"train pairs {len(P_tr):,} ({P_tr['trip_id'].nunique()} trips) | test pairs {len(P_te):,}", flush=True)

    res_dir = config.RESULTS_DIR / "models"
    (res_dir / "runs").mkdir(parents=True, exist_ok=True)
    pred_dir = config.CACHE_DIR / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    table_path = res_dir / "model_comparison.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()

    for name in args.models:
        try:
            table = run_one(name, P_tr, Q_tr, P_te, Q_te, folds, res_dir, pred_dir, table, table_path)
        except Exception as exc:  # keep going: one failing model must not cost the others
            import traceback
            traceback.print_exc()
            print(f"!!! {name} failed: {exc!r}", flush=True)

    cols = ["model", "cv_macro_f1_mean", "cv_top1", "cv_log_loss", "test_top1", "test_macro_f1",
            "test_log_loss", "test_route_continuity", "infer_ms_per_1k_points"]
    print("\n" + table[cols].round(4).to_string(index=False))


def run_one(name, P_tr, Q_tr, P_te, Q_te, folds, res_dir, pred_dir, table, table_path):
    if True:
        spec = get_spec(name)
        rng = np.random.default_rng(config.SEED)
        print(f"\n=== {name} ({spec.family}) ===", flush=True)
        oof = np.zeros(len(P_tr))
        fold_rows = []
        for f, (tr_idx, va_idx) in enumerate(folds):
            P_a, P_b = P_tr.iloc[tr_idx], P_tr.iloc[va_idx]
            prob, _, fit_s, pred_s = fit_predict(spec, P_a, P_b, config.SEED, rng)
            oof[va_idx] = prob
            m = metrics(P_b, Q_tr, prob)
            m.update(fold=f, fit_s=fit_s, predict_s=pred_s)
            fold_rows.append(m)
            print(f"  fold {f}: top1 {m['top1']:.4f}  macroF1 {m['macro_f1']:.4f}  logloss {m['log_loss']:.4f}"
                  f"  fit {fit_s:.1f}s", flush=True)
        folds_df = pd.DataFrame(fold_rows)
        cv = metrics(P_tr, Q_tr, oof)

        prob_te, model, fit_s, pred_s = fit_predict(spec, P_tr, P_te, config.SEED, rng)
        te = metrics(P_te, Q_te, prob_te)
        cont = continuity(P_te, prob_te, "student_test")
        n_te_points = len(Q_te)
        print(f"  CV  : top1 {cv['top1']:.4f}  macroF1 {folds_df['macro_f1'].mean():.4f}±{folds_df['macro_f1'].std():.4f}"
              f"  logloss {cv['log_loss']:.4f}")
        print(f"  TEST: top1 {te['top1']:.4f}  macroF1 {te['macro_f1']:.4f}  logloss {te['log_loss']:.4f}"
              f"  continuity {cont:.4f}  fit {fit_s:.1f}s  predict {1000 * pred_s / n_te_points * 1000:.1f} ms/1k pts", flush=True)

        pd.DataFrame({"trip_id": P_tr["trip_id"], "obs": P_tr["obs"], "arc_id": P_tr["arc_id"], "prob": oof}
                     ).to_parquet(pred_dir / f"{name}_oof.parquet", index=False)
        pd.DataFrame({"trip_id": P_te["trip_id"], "obs": P_te["obs"], "arc_id": P_te["arc_id"], "prob": prob_te}
                     ).to_parquet(pred_dir / f"{name}_test.parquet", index=False)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        row = {
            "model": name, "family": spec.family, "config": json.dumps(spec.params, default=str),
            "train_point_frac": spec.train_point_frac,
            "cv_macro_f1_mean": folds_df["macro_f1"].mean(), "cv_macro_f1_std": folds_df["macro_f1"].std(),
            "cv_top1": cv["top1"], "cv_top1_fold_std": folds_df["top1"].std(), "cv_log_loss": cv["log_loss"],
            "test_top1": te["top1"], "test_macro_f1": te["macro_f1"], "test_precision": te["precision"],
            "test_recall": te["recall"], "test_log_loss": te["log_loss"], "test_route_continuity": cont,
            "fit_s_full": fit_s, "infer_ms_per_1k_points": 1e6 * pred_s / n_te_points,
            "seed": config.SEED, "run": stamp,
        }
        save_model(spec, model, row)
        run = {"row": row, "folds": fold_rows, "cv_oof": cv, "test": te,
               "features": spec.features or FEATURES, "n_folds": N_FOLDS, "candidate_radius": config.CAND_RADIUS,
               "candidate_k": config.CAND_K, "label_max_dist": config.LABEL_MAX_DIST}
        if spec.epoch_history:
            if hasattr(model, "history_"):  # torch: per-epoch train / validation loss
                run["history"] = model.history_
                run["best_epoch"] = model.best_epoch_
            else:  # sklearn MLP
                mlp = model[-1]
                run["history"] = {"train_loss": list(map(float, mlp.loss_curve_)),
                                  "val_accuracy": list(map(float, getattr(mlp, "validation_scores_", []) or []))}
        (res_dir / "runs" / f"{name}_{stamp}.json").write_text(json.dumps(run, indent=2, default=str))
        table = pd.concat([table[table.get("model", pd.Series(dtype=str)) != name] if len(table) else table,
                           pd.DataFrame([row])], ignore_index=True)
        table.to_csv(table_path, index=False)
    return table


if __name__ == "__main__":
    main()
