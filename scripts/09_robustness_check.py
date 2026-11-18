"""Robustness check: the evaluator's trips are unseen, so the pipeline must not crash on surprises.

Builds edge-case trips from ONE student-test trip in a temporary folder and runs the real
predict_trip / score_trip on each, with an explicit expectation per case:
    flat_layout      all files in one folder instead of a sub-folder per trip
    no_route         .route missing -> predict, but metrics are null
    shuffled_time    rows of .track out of time order -> same result as sorted
    far_from_roads   track moved ~5 km -> roads in the corridor, none within the radius
    empty_corridor   track moved ~111 km -> the cropped road network is empty
    single_point     a one-point trip
Every response must also serialise as strict JSON (no NaN / inf), as FastAPI requires.
Exits with status 1 if any case fails.
Usage: python scripts/09_robustness_check.py [--model logreg] [--trip 00000052]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from pathsnap import config
from pathsnap.features import prepare_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import predict_trip, score_trip


def write_track(path: Path, track: np.ndarray) -> None:
    np.savetxt(path, track, fmt=["%.6f", "%.6f", "%.0f"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="saved model name (default: the exported final model)")
    ap.add_argument("--trip", default="00000052")
    args = ap.parse_args()
    src = config.DATA_ROOT / "student_test" / args.trip
    tid = args.trip
    root = Path(tempfile.mkdtemp(prefix="pathsnap_robust_"))
    track = np.loadtxt(src / f"{tid}.track", ndmin=2)
    cases = {}

    flat = root / "flat_layout"
    flat.mkdir()
    for f in src.iterdir():
        shutil.copy(f, flat / f.name)
    cases["flat_layout"] = list_trips(flat)[0]

    for name in ("no_route", "shuffled_time", "far_from_roads", "empty_corridor", "single_point"):
        d = root / name / tid
        d.mkdir(parents=True)
        for ext in (".nodes", ".arcs", ".route"):
            if not (name == "no_route" and ext == ".route"):
                shutil.copy(src / f"{tid}{ext}", d / f"{tid}{ext}")
        t = track.copy()
        if name == "shuffled_time":
            t = t[np.random.default_rng(0).permutation(len(t))]
        elif name == "far_from_roads":
            t[:, 1] += 0.045  # ~5 km north
        elif name == "empty_corridor":
            t[:, 1] += 1.0  # ~111 km north: nothing of the network within the corridor
        elif name == "single_point":
            t = t[:1]
        write_track(d / f"{tid}.track", t)
        cases[name] = list_trips(root / name)[0]

    results, failures = {}, []
    for name, trip_dir in cases.items():
        try:
            pred = predict_trip(trip_dir, model=args.model)
            m = score_trip(trip_dir, pred)
            json.dumps(pred, allow_nan=False)  # what FastAPI's JSONResponse does
            json.dumps(m, allow_nan=False)
            results[name] = (pred, m)
            n = pred["n_points"]
            if name == "flat_layout":
                assert m["top1_accuracy"] is not None and m["top1_accuracy"] > 0.8, m["top1_accuracy"]
            elif name == "no_route":
                assert m["top1_accuracy"] is None and "note" in m, m
            elif name == "shuffled_time":
                assert m["top1_accuracy"] == results["flat_layout"][1]["top1_accuracy"], m["top1_accuracy"]
            elif name == "far_from_roads":
                assert pred["n_unmatched"] == n and pred["route_arcs"] == [], pred["n_unmatched"]
            elif name == "empty_corridor":
                ctx = prepare_trip(load_trip(trip_dir), config.CAND_RADIUS, config.CAND_K)
                assert len(ctx.net.arc_ids) == 0, f"corridor not empty: {len(ctx.net.arc_ids)} arcs"
                assert pred["n_unmatched"] == n and pred["route_arcs"] == [], pred["n_unmatched"]
                assert pred["route_continuity"] is None, pred["route_continuity"]
            elif name == "single_point":
                assert n == 1, n
            print(f"{name:15s} OK   points {n:4d}  unmatched {pred['n_unmatched']:4d}  route arcs "
                  f"{len(pred['route_arcs']):4d}  pieces {pred['n_route_pieces']}  top1 {m['top1_accuracy']}"
                  f"  continuity {pred['route_continuity']}  {m.get('note', '')}")
        except Exception:
            failures.append(name)
            print(f"{name:15s} FAIL")
            traceback.print_exc()
    shutil.rmtree(root, ignore_errors=True)
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
