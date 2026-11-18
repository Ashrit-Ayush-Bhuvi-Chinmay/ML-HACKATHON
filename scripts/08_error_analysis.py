"""Where do classifier errors come from? (student-test or train OOF predictions)

Categories for a reliably labelled point whose top candidate is wrong:
    twin        predicted arc is the reverse direction of the reference arc (same road)
    adjacent    predicted arc shares a node with the reference arc (junction boundary)
    other       a different road (parallel street, wrong branch, ...)
    no_cands    reference not among candidates / point has no candidates
Also splits accuracy by speed (stopped vehicles have no usable heading).

Usage: python scripts/08_error_analysis.py --model lightgbm --split student_test
Output: results/errors/errors_<model>_<split>.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.evaluate import best_per_point
from pathsnap.io import load_trip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--split", default="student_test", choices=["student_test", "train"])
    args = ap.parse_args()
    suffix = "test" if args.split == "student_test" else "oof"
    preds = pd.read_parquet(config.CACHE_DIR / "preds" / f"{args.model}_{suffix}.parquet")
    pairs = pd.read_parquet(config.CACHE_DIR / "dataset" / f"{args.split}_pairs.parquet",
                            columns=["trip_id", "obs", "speed", "heading_valid", "dist", "n_cands"])
    points = pd.read_parquet(config.CACHE_DIR / "dataset" / f"{args.split}_points.parquet")
    rows = []
    for trip_id, g in preds.groupby("trip_id", sort=False):
        arcs = load_trip(config.DATA_ROOT / args.split / trip_id).arcs
        idx = g.index.to_numpy()
        k, best = best_per_point(g["obs"].to_numpy(), g["prob"].to_numpy())
        pred_arc = g["arc_id"].to_numpy()[best]
        pq = points[points["trip_id"] == trip_id].set_index("obs")
        pq = pq.loc[k]
        feat = pairs.loc[idx[best], ["speed", "heading_valid"]].to_numpy()
        ref = pq["ref_arc"].to_numpy()
        rel = pq["reliable"].to_numpy()
        inset = pq["ref_in_cands"].to_numpy()
        correct = pred_arc == ref
        pu, pv = arcs[pred_arc, 0], arcs[pred_arc, 1]
        ru, rv = arcs[ref, 0], arcs[ref, 1]
        twin = (pu == rv) & (pv == ru)
        adjacent = ~twin & ((pu == ru) | (pu == rv) | (pv == ru) | (pv == rv))
        cat = np.where(correct, "correct", np.where(~inset, "no_cands", np.where(twin, "twin",
                       np.where(adjacent, "adjacent", "other"))))
        rows.append(pd.DataFrame({"trip_id": trip_id, "obs": k, "reliable": rel, "category": cat,
                                  "speed": feat[:, 0], "heading_valid": feat[:, 1]}))
    df = pd.concat(rows, ignore_index=True)
    df = df[df["reliable"]]
    out = config.RESULTS_DIR / "errors"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"errors_{args.model}_{args.split}.csv", index=False)

    print(f"{args.model} on {args.split}: {len(df):,} reliable points with candidates")
    print((df["category"].value_counts(normalize=True) * 100).round(2).to_string())
    df["speed_bin"] = pd.cut(df["speed"], [-0.1, 1, 3, 8, 15, 100], labels=["<1", "1-3", "3-8", "8-15", ">15 m/s"])
    t = df.groupby("speed_bin", observed=True).agg(points=("category", "size"),
                                                   accuracy=("category", lambda s: (s == "correct").mean()),
                                                   twin=("category", lambda s: (s == "twin").mean()),
                                                   adjacent=("category", lambda s: (s == "adjacent").mean()),
                                                   other=("category", lambda s: (s == "other").mean()))
    print("\nby speed:\n" + t.round(4).to_string())
    print("\nby trip (accuracy):")
    print(df.groupby("trip_id")["category"].apply(lambda s: (s == "correct").mean()).round(4).to_string())


if __name__ == "__main__":
    main()
