"""Step 4 driver: build the (GPS point, candidate arc) feature tables.

Outputs (git-ignored, regenerate with this script)
    cache/dataset/<split>_pairs.parquet    one row per candidate: trip_id, obs, arc_id, features, y
    cache/dataset/<split>_points.parquet   one row per GPS point: reliable label?, n_cands, ref in set?
Summary: results/dataset/dataset_summary.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.features import pair_features, prepare_trip
from pathsnap.io import list_trips, load_trip


def main() -> None:
    bad = config.bad_reference_trips()
    out = config.CACHE_DIR / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for split in config.SPLITS:
        pairs, points = [], []
        for trip_dir in list_trips(config.DATA_ROOT / split):
            if (split, trip_dir.name) in bad:
                print(f"skip {split}/{trip_dir.name} (unusable reference)")
                continue
            t0 = time.perf_counter()
            trip = load_trip(trip_dir)
            lab = np.load(config.CACHE_DIR / "labels" / split / f"{trip.trip_id}.npz")
            ctx = prepare_trip(trip, config.CAND_RADIUS, config.CAND_K)
            df = pair_features(ctx)
            secs = time.perf_counter() - t0

            ref = lab["arc_id"]
            reliable = lab["dist"] <= config.LABEL_MAX_DIST
            df["y"] = (df["arc_id"].to_numpy() == ref[df["obs"].to_numpy()]).astype(np.int8)
            df["reliable"] = reliable[df["obs"].to_numpy()]
            df.insert(0, "trip_id", trip.trip_id)
            pairs.append(df)

            n = trip.n_points
            n_c = np.bincount(df["obs"], minlength=n)
            hit = np.bincount(df["obs"][df["y"] == 1], minlength=n) > 0
            points.append(pd.DataFrame({"trip_id": trip.trip_id, "obs": np.arange(n), "reliable": reliable,
                                        "n_cands": n_c, "ref_in_cands": hit, "ref_arc": ref}))
            print(f"{split:>12} {trip.trip_id}  points {n:>6}  pairs {len(df):>7}  {secs:5.1f}s", flush=True)

        P = pd.concat(pairs, ignore_index=True)
        Q = pd.concat(points, ignore_index=True)
        P.to_parquet(out / f"{split}_pairs.parquet", index=False)
        Q.to_parquet(out / f"{split}_points.parquet", index=False)
        rel = Q["reliable"]
        summary.append({"split": split, "trips": Q["trip_id"].nunique(), "points": len(Q),
                        "reliable_points": int(rel.sum()), "pairs": len(P),
                        "pairs_per_point": len(P) / len(Q), "positives": int(P["y"].sum()),
                        "miss_rate_reliable": float(1 - Q.loc[rel, "ref_in_cands"].mean()),
                        "empty_rate": float((Q["n_cands"] == 0).mean())})
    s = pd.DataFrame(summary)
    res = config.RESULTS_DIR / "dataset"
    res.mkdir(parents=True, exist_ok=True)
    s.to_csv(res / "dataset_summary.csv", index=False)
    print("\n" + s.to_string(index=False))


if __name__ == "__main__":
    main()
