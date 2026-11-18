"""Clean inference timing for every saved model (run on an otherwise idle machine).

The comparison script times models while other jobs may share the CPU; this re-measures
prediction cost under equal conditions: each saved model scores the full student-test candidate
table (169k rows) three times, and the fastest run is kept. Also times the complete pipeline
(load -> candidates -> features -> model -> HMM) per student-test trip for the final model.

Output: results/models/inference_timing.csv, results/pipeline/latency_default.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.io import list_trips
from pathsnap.pipeline import ARTIFACT_PATH, MODELS_DIR, _score_candidates, load_model, predict_trip


def main() -> None:
    T = pd.read_parquet(config.CACHE_DIR / "dataset" / "student_test_pairs.parquet")
    obs_key = (pd.factorize(T["trip_id"])[0].astype(np.int64) * 1_000_000 + T["obs"].to_numpy())
    n_points = T.groupby("trip_id")["obs"].nunique().sum()
    rows = []
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        b = load_model(path.stem)
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            _score_candidates(b, T, obs_key)
            best = min(best, time.perf_counter() - t0)
        rows.append({"model": path.stem, "rows": len(T), "points": int(n_points), "seconds": best,
                     "ms_per_1k_points": 1e6 * best / n_points})
        print(f"{path.stem:20s} {1e6 * best / n_points:8.2f} ms per 1k points", flush=True)
    out = config.RESULTS_DIR / "models"
    pd.DataFrame(rows).to_csv(out / "inference_timing.csv", index=False)

    if ARTIFACT_PATH.exists():
        from pathsnap.pipeline import score_trip

        lat = []
        trips = list_trips(config.DATA_ROOT / "student_test")
        t0 = time.perf_counter()
        predict_trip(trips[0])  # cold call: includes loading the model bundle
        cold_ms = 1000 * (time.perf_counter() - t0)
        for trip_dir in trips:
            t0 = time.perf_counter()
            pred = predict_trip(trip_dir)
            wall_pred = 1000 * (time.perf_counter() - t0)
            t0 = time.perf_counter()
            score_trip(trip_dir, pred)
            wall_score = 1000 * (time.perf_counter() - t0)
            lat.append({"trip_id": pred["trip_id"], "n_points": pred["n_points"], "wall_ms_predict": wall_pred,
                        "wall_ms_score": wall_score, "cold_first_call_ms": cold_ms,
                        **{f"stage_ms_{k}": v for k, v in pred["timings_ms"].items()}})
        df = pd.DataFrame(lat)
        (config.RESULTS_DIR / "pipeline").mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RESULTS_DIR / "pipeline" / "latency_default.csv", index=False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
