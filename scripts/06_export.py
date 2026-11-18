"""Step 8: train the chosen model on all train trips and export the inference bundle, then run
the full pipeline (exactly what the API runs) on every student-test trip as a rehearsal.

Usage: python scripts/06_export.py --model lightgbm [--params-json results/tuning/best_lightgbm.json]
Output: artifacts/pathsnap_model.joblib, results/pipeline/rehearsal_<model>.csv
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.evaluate import point_keys
from pathsnap.features import FEATURES
from pathsnap.hmm import HMMParams
from pathsnap.io import list_trips
from pathsnap.models import SPECS
from pathsnap.pipeline import ARTIFACT_PATH, predict_trip, score_trip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--params-json", default=None, help="tuned hyperparameters (overrides the spec)")
    ap.add_argument("--hmm-summary", default=None, help="results/hmm/summary_<model>.csv with tuned decoder params")
    args = ap.parse_args()

    spec = SPECS[args.model]
    params = dict(spec.params)
    if args.params_json:
        params.update(json.loads(open(args.params_json).read()))
    hmm_path = args.hmm_summary or (config.RESULTS_DIR / "hmm" / f"summary_{args.model}.csv")
    try:
        hmm_params = json.loads(pd.read_csv(hmm_path)["params"].iloc[0])
    except FileNotFoundError:
        hmm_params = HMMParams().as_dict()
    print("hmm params:", hmm_params)

    P = pd.read_parquet(config.CACHE_DIR / "dataset" / "train_pairs.parquet")
    P = P[P["reliable"]].reset_index(drop=True)
    X, y = P[FEATURES].to_numpy(), P["y"].to_numpy()
    t0 = time.perf_counter()
    model = spec.build(params, config.SEED)
    if hasattr(model, "fit_grouped"):
        codes = {t: i for i, t in enumerate(sorted(P["trip_id"].unique()))}
        model.fit_grouped(X, y, point_keys(P["trip_id"].map(codes).to_numpy(), P["obs"].to_numpy()))
    else:
        model.fit(X, y)
    fit_s = time.perf_counter() - t0
    bundle = {
        "model": model, "model_name": args.model, "model_params": params, "features": FEATURES,
        "candidate_radius": config.CAND_RADIUS, "candidate_k": config.CAND_K, "hmm_params": hmm_params,
        "label_max_dist": config.LABEL_MAX_DIST, "seed": config.SEED,
        "trained_on": f"train split, {P['trip_id'].nunique()} trips, {len(P):,} candidate rows",
        "created": datetime.now().isoformat(timespec="seconds"), "fit_seconds": fit_s,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)
    print(f"saved {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size / 1e6:.1f} MB), fit {fit_s:.1f}s")

    rows = []
    for trip_dir in list_trips(config.DATA_ROOT / "student_test"):
        pred = predict_trip(trip_dir, bundle)
        m = score_trip(trip_dir, pred)
        rows.append({"trip_id": pred["trip_id"], "n_points": pred["n_points"], **m,
                     **{f"ms_{k}": v for k, v in pred["timings_ms"].items()}})
        print(f"{pred['trip_id']}  top1 {m['top1_accuracy']:.4f}  routeF1 {m['route_f1']:.4f}"
              f"  continuity {m['route_continuity']:.4f}  {pred['timings_ms']['total']:.0f} ms", flush=True)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"rehearsal_{args.model}.csv", index=False)
    w = df["n_points_scored"] / df["n_points_scored"].sum()
    print(f"\npoint-weighted top1 {np.sum(w * df['top1_accuracy']):.4f} | mean route F1 {df['route_f1'].mean():.4f}"
          f" | mean continuity {df['route_continuity'].mean():.4f} | max latency {df['ms_total'].max():.0f} ms")


if __name__ == "__main__":
    main()
