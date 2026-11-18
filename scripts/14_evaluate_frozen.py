"""Final evaluation of the FROZEN system on the evaluator test folder (one recorded pass).

Refuses to run unless artifacts/pathsnap_model.joblib matches the SHA-256 recorded in
docs/FREEZE.md. Nothing here fits, tunes or selects anything: the final system (default model +
HMM) is evaluated, plus two reference variants for the report only (classifier without HMM, and
the distance-only baseline). Results must not be used to change the model, features or decoder.

Usage: python scripts/14_evaluate_frozen.py --eval-dir "<evaluator_test folder>"
Output: results/evaluation/evaluator_per_trip.csv, results/evaluation/evaluator_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.io import list_trips
from pathsnap.pipeline import ARTIFACT_PATH, predict_trip, score_trip

VARIANTS = [  # (label, model, use_hmm)
    ("final_system", None, True),
    ("final_model_no_hmm", None, False),
    ("distance_baseline_no_hmm", "distance_gaussian", False),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    args = ap.parse_args()

    frozen = re.search(r"pathsnap_model\.joblib`\s+SHA-256 `([0-9a-f]{64})`",
                       (config.REPO_ROOT / "docs" / "FREEZE.md").read_text(encoding="utf-8")).group(1)
    actual = sha256(ARTIFACT_PATH)
    if actual != frozen:
        raise SystemExit(f"model bundle does not match the freeze record ({actual} != {frozen}); refusing to run")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, capture_output=True,
                            text=True).stdout.strip()

    rows = []
    for trip_dir in list_trips(Path(args.eval_dir)):
        for label, model, use_hmm in VARIANTS:
            try:
                t0 = time.perf_counter()
                pred = predict_trip(trip_dir, model=model, use_hmm=use_hmm)
                t_pred = time.perf_counter() - t0
                m = score_trip(trip_dir, pred)
                rows.append({"trip_id": trip_dir.name, "variant": label, "n_points": pred["n_points"],
                             "n_unmatched": pred["n_unmatched"], "n_route_pieces": pred["n_route_pieces"],
                             "gps_looks_offset": pred["offset_correction"].get("gps_looks_offset"),
                             **m, "wall_ms_predict": round(1000 * t_pred), "error": None})
            except Exception as exc:  # recorded, not hidden
                traceback.print_exc()
                rows.append({"trip_id": trip_dir.name, "variant": label, "error": repr(exc)})
            r = rows[-1]
            print(f"{r['trip_id']} {label:26s} top1 {r.get('top1_accuracy')}  seqF1 {r.get('sequence_f1')}  "
                  f"covF1 {r.get('coverage_f1')}  cont {r.get('route_continuity')}  "
                  f"{r.get('wall_ms_predict')} ms  {r.get('note') or r.get('error') or ''}", flush=True)

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "evaluator_per_trip.csv", index=False)

    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "eval_dir": str(args.eval_dir),
               "model_sha256": actual, "code_commit": commit, "n_trips": int(df["trip_id"].nunique()),
               "n_errors": int(df["error"].notna().sum()), "variants": {}}
    for label, _, _ in VARIANTS:
        d = df[(df["variant"] == label) & df["error"].isna()]
        if d.empty:
            continue
        w = d["n_points_scored"] / d["n_points_scored"].sum()
        summary["variants"][label] = {
            "top1_point_weighted": float((w * d["top1_accuracy"]).sum()),
            "top1_mean_over_trips": float(d["top1_accuracy"].mean()),
            "sequence_f1_mean": float(d["sequence_f1"].mean()),
            "coverage_f1_mean": float(d["coverage_f1"].mean()),
            "route_continuity_mean": float(d["route_continuity"].dropna().mean()),
            "route_mismatch_mean": float(d["route_mismatch"].mean()),
            "max_predict_ms": int(d["wall_ms_predict"].max()),
        }
    (out / "evaluator_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
