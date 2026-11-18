"""Robustness check: the evaluator's trips are unseen, so the pipeline must not crash on surprises.

Builds edge-case trips from ONE student-test trip in a temporary folder and runs the real
predict_trip / score_trip on each:
    flat_layout      all files in one folder instead of a sub-folder per trip
    no_route         .route missing -> predict, but metrics are null
    shuffled_time    rows of .track out of time order
    far_from_roads   track moved 5 km away -> no candidates at all
    single_point     a one-point trip
Usage: python scripts/09_robustness_check.py [--model logreg] [--trip 00000052]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

from pathsnap import config
from pathsnap.io import list_trips
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

    for name in ("no_route", "shuffled_time", "far_from_roads", "single_point"):
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
        elif name == "single_point":
            t = t[:1]
        write_track(d / f"{tid}.track", t)
        cases[name] = list_trips(root / name)[0]

    ok = True
    for name, trip_dir in cases.items():
        try:
            pred = predict_trip(trip_dir, model=args.model)
            m = score_trip(trip_dir, pred)
            print(f"{name:15s} OK   points {pred['n_points']:4d}  unmatched {pred['n_unmatched']:4d}"
                  f"  route arcs {len(pred['route_arcs']):4d}  top1 {m['top1_accuracy']}  {m.get('note', '')}")
        except Exception:
            ok = False
            print(f"{name:15s} FAIL")
            traceback.print_exc()
    shutil.rmtree(root, ignore_errors=True)
    print("ALL PASSED" if ok else "SOME CASES FAILED")


if __name__ == "__main__":
    main()
