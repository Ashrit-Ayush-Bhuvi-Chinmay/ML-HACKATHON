"""Validate the systematic-GPS-offset correction on the one supplied trip that needs it (train 54:
whole track ~117 m off the road network), and confirm it does nothing on normal trips.

Runs predict_trip + score_trip with and without the correction.
Usage: python scripts/11_offset_check.py [--model logreg]
Output: results/pipeline/offset_check.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from pathsnap import config
from pathsnap.pipeline import predict_trip, score_trip

TRIPS = [("train", "00000054"), ("train", "00000033"), ("student_test", "00000052"), ("student_test", "00000076")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg")
    args = ap.parse_args()
    rows = []
    for split, tid in TRIPS:
        trip_dir = config.DATA_ROOT / split / tid
        for corr in (False, True):
            pred = predict_trip(trip_dir, model=args.model, offset_correction=corr)
            m = score_trip(trip_dir, pred)
            off = pred["offset_correction"]
            rows.append({"split": split, "trip_id": tid, "correction": corr, **off,
                         **{k: m[k] for k in ("top1_accuracy", "route_f1", "route_precision", "route_recall",
                                              "n_points_scored", "reference_median_dist_m")}})
            print(f"{split}/{tid} correction={corr!s:5}  offset {off}  top1 {m['top1_accuracy']}  "
                  f"routeF1 {m['route_f1']}  scored {m['n_points_scored']}  ref_med {m['reference_median_dist_m']}",
                  flush=True)
    out = config.RESULTS_DIR / "pipeline"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "offset_check.csv", index=False)


if __name__ == "__main__":
    main()
