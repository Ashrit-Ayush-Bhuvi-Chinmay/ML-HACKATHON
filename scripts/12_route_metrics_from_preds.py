"""Recompute classifier-only ROUTE metrics for every model from its saved student-test
predictions (no retraining), so the leaderboard uses one consistent set of definitions:

    test_strict_adjacency       head -> tail node sharing (pessimistic, kept for reference)
    test_plausible_continuity   agreed metric: reachable within the distance budget, pooled
                                over transitions, gaps excluded
    test_coverage_f1            length-weighted arc-set overlap (order-insensitive)
    test_sequence_f1            length-weighted LCS of arc sequences (order/lap-aware)

Updates results/models/model_comparison.csv and artifacts/models/registry.json in place and
writes results/models/route_metrics_classifier_only.csv (per model x trip).
Usage: python scripts/12_route_metrics_from_preds.py [--models a b ...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.evaluate import arc_lengths_m, best_per_point, route_scores, strict_adjacency
from pathsnap.features import prepare_trip
from pathsnap.hmm import HMMParams, RoadGraph, build_route, plausible_continuity, segment_ids
from pathsnap.io import load_route, load_trip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    table_path = config.RESULTS_DIR / "models" / "model_comparison.csv"
    table = pd.read_csv(table_path)
    models = args.models or [m for m in table["model"] if (config.CACHE_DIR / "preds" / f"{m}_test.parquet").exists()]
    params = HMMParams()

    # trip contexts are shared by all models: build once
    first = pd.read_parquet(config.CACHE_DIR / "preds" / f"{models[0]}_test.parquet")
    trips = {}
    for trip_id in first["trip_id"].unique():
        trip_dir = config.DATA_ROOT / "student_test" / trip_id
        trip = load_trip(trip_dir)
        ctx = prepare_trip(trip, config.CAND_RADIUS, config.CAND_K)
        ref = load_route(trip_dir).tolist()
        trips[trip_id] = (trip, ctx, RoadGraph(ctx, params.sp_limit), segment_ids(ctx, params), ref)

    rows = []
    for model in models:
        preds = pd.read_parquet(config.CACHE_DIR / "preds" / f"{model}_test.parquet").groupby("trip_id", sort=False)
        for trip_id, (trip, ctx, graph, seg, ref) in trips.items():
            g = preds.get_group(trip_id)
            assert np.array_equal(g["obs"].to_numpy(), ctx.cands.obs), f"{model}/{trip_id}: candidate order"
            point_rows = np.full(trip.n_points, -1, dtype=np.int64)
            k, b = best_per_point(ctx.cands.obs, g["prob"].to_numpy())
            point_rows[k] = b
            cont, n_tr = plausible_continuity(ctx, graph, point_rows, seg)
            pieces = build_route(ctx, graph, point_rows, seg)
            pred_seq = [a for p in pieces for a in p]
            sc = route_scores(pred_seq, ref, arc_lengths_m(trip, ctx.proj, pred_seq + ref))
            arc_per_point = ctx.cands.arc_id[point_rows[point_rows >= 0]]
            rows.append({"model": model, "trip_id": trip_id, "continuity": cont, "n_transitions": n_tr,
                         "strict_adjacency": strict_adjacency(arc_per_point, trip.arcs), **sc})
        print(f"{model}: done", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(config.RESULTS_DIR / "models" / "route_metrics_classifier_only.csv", index=False)

    def pooled(g):
        n = g["n_transitions"].sum()
        return float((g["continuity"].fillna(0) * g["n_transitions"]).sum() / n) if n else None

    summ = per.groupby("model").apply(lambda g: pd.Series({
        "test_plausible_continuity": pooled(g),
        "test_strict_adjacency": g["strict_adjacency"].mean(),
        "test_coverage_f1": g["coverage_f1"].mean(),
        "test_sequence_f1": g["sequence_f1"].mean()}), include_groups=False)

    # the old column held strict adjacency under a misleading name; it is replaced by the four below
    table = table.drop(columns=[c for c in ("test_route_continuity",) if c in table.columns])
    for col in summ.columns:  # update only the recomputed models' rows, keep the others as they are
        if col not in table.columns:
            table[col] = np.nan
        for model, val in summ[col].items():
            table.loc[table["model"] == model, col] = val
    table.to_csv(table_path, index=False)

    reg_path = config.REPO_ROOT / "artifacts" / "models" / "registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        for name, info in reg.items():
            info.pop("test_route_continuity", None)
            if name in summ.index:
                info.update({k: (None if pd.isna(v) else float(v)) for k, v in summ.loc[name].items()})
        reg_path.write_text(json.dumps(reg, indent=2, default=str))
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
