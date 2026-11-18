"""Step 2 driver: derive a reference arc for every GPS observation of every supplied trip.

Outputs
    cache/labels/<split>/<trip_id>.npz       per-point arrays (see pathsnap.labels.label_trip)
    results/labels/labels_summary.csv        one row per trip: label quality statistics

Also runs a distance-only alignment (w_heading = 0) as an ablation, to measure what the
heading term changes.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.labels import label_trip

MIN_SPEED = 2.0


def summarise(split, trip, route, lab, lab0, secs):
    d = lab["dist"]
    moving = lab["speed"] >= MIN_SPEED
    return {
        "split": split,
        "trip_id": trip.trip_id,
        "n_points": trip.n_points,
        "n_route_arcs": len(route),
        "n_network_arcs": len(trip.arcs),
        "dist_median_m": float(np.median(d)),
        "dist_p90_m": float(np.percentile(d, 90)),
        "dist_p99_m": float(np.percentile(d, 99)),
        "dist_max_m": float(d.max()),
        "frac_gt_20m": float((d > 20).mean()),
        "frac_gt_50m": float((d > 50).mean()),
        "frac_gt_100m": float((d > 100).mean()),
        "route_coverage": len(np.unique(lab["route_pos"])) / len(route),
        "dir_agree_moving": float((lab["cos_dir"][moving] > 0).mean()) if moving.any() else np.nan,
        "dir_agree_moving_no_heading": float((lab0["cos_dir"][moving] > 0).mean()) if moving.any() else np.nan,
        "frac_label_changed_by_heading": float((lab["arc_id"] != lab0["arc_id"]).mean()),
        "seconds": secs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=list(config.SPLITS))
    ap.add_argument("--w-heading", type=float, default=25.0)
    ap.add_argument("--cap", type=float, default=150.0)
    args = ap.parse_args()

    rows, all_d = [], {s: [] for s in args.splits}
    for split in args.splits:
        out_dir = config.CACHE_DIR / "labels" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for trip_dir in list_trips(config.DATA_ROOT / split):
            t0 = time.perf_counter()
            trip = load_trip(trip_dir)
            route = load_route(trip_dir)
            lab = label_trip(trip, route, w_heading=args.w_heading, cap=args.cap, min_speed=MIN_SPEED)
            lab0 = label_trip(trip, route, w_heading=0.0, cap=args.cap, min_speed=MIN_SPEED)
            secs = time.perf_counter() - t0
            np.savez_compressed(out_dir / f"{trip.trip_id}.npz", **lab)
            row = summarise(split, trip, route, lab, lab0, secs)
            rows.append(row)
            all_d[split].append(lab["dist"])
            print(f"{split:>12} {trip.trip_id}  n={row['n_points']:>6}  median={row['dist_median_m']:6.1f} m"
                  f"  p90={row['dist_p90_m']:6.1f} m  >50m={row['frac_gt_50m']:6.2%}"
                  f"  dir={row['dir_agree_moving']:.3f} (no-heading {row['dir_agree_moving_no_heading']:.3f})"
                  f"  {secs:5.1f}s", flush=True)

    df = pd.DataFrame(rows)
    res_dir = config.RESULTS_DIR / "labels"
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "labels_summary.csv", index=False)

    print("\n=== label distance (metres from GPS point to its reference arc) ===")
    for split, parts in all_d.items():
        d = np.concatenate(parts)
        q = np.percentile(d, [50, 75, 90, 95, 99])
        print(f"{split:>12}: n={len(d)}  p50={q[0]:.1f} p75={q[1]:.1f} p90={q[2]:.1f} p95={q[3]:.1f} "
              f"p99={q[4]:.1f}  >20m={np.mean(d > 20):.2%} >50m={np.mean(d > 50):.2%} >100m={np.mean(d > 100):.2%}")
    w = df["n_points"] / df["n_points"].sum()
    print(f"direction agreement (moving points, weighted): with heading {np.nansum(w * df['dir_agree_moving']):.3f}"
          f" | distance only {np.nansum(w * df['dir_agree_moving_no_heading']):.3f}")
    print(f"labels changed by the heading term: {np.sum(w * df['frac_label_changed_by_heading']):.2%} of points")
    print(f"mean route coverage: {df['route_coverage'].mean():.3f}")
    print("worst trips by p90 distance:")
    print(df.sort_values("dist_p90_m", ascending=False).head(8)[
        ["split", "trip_id", "n_points", "dist_median_m", "dist_p90_m", "frac_gt_50m", "route_coverage"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
