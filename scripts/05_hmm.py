"""Step 7 driver: classifier-only vs classifier + HMM/Viterbi.

1. Tune the decoder (beta, emission weight) on TRAIN trips using out-of-fold classifier
   probabilities (so the decoder never sees a probability the classifier was trained on).
2. With the chosen parameters, decode the student-test trips and compare
     classifier-only (argmax per point) | classifier + HMM | distance-only HMM (classic)
   on Top-1, plausible route continuity and route-level precision / recall / F1 / mismatch.

Outputs: results/hmm/tuning.csv, results/hmm/per_trip.csv, results/hmm/summary.csv
Usage: python scripts/05_hmm.py --model lightgbm [--tune-trips 30]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.evaluate import arc_lengths_m, best_per_point, route_scores
from pathsnap.features import prepare_trip
from pathsnap.hmm import HMMParams, build_route, plausible_continuity, segment_ids, viterbi
from pathsnap.io import list_trips, load_route, load_trip


def trip_inputs(split: str, trip_id: str, preds: pd.DataFrame):
    trip = load_trip(config.DATA_ROOT / split / trip_id)
    ctx = prepare_trip(trip, config.CAND_RADIUS, config.CAND_K)
    p = preds.get_group(trip_id)
    if not (np.array_equal(p["obs"].to_numpy(), ctx.cands.obs) and np.array_equal(p["arc_id"].to_numpy(), ctx.cands.arc_id)):
        raise RuntimeError(f"{trip_id}: prediction rows do not match regenerated candidates")
    lab = np.load(config.CACHE_DIR / "labels" / split / f"{trip_id}.npz")
    return trip, ctx, p["prob"].to_numpy(), lab


def argmax_rows(ctx, prob) -> np.ndarray:
    rows = np.full(len(ctx.px), -1, dtype=np.int64)
    if len(ctx.cands.obs):
        k, idx = best_per_point(ctx.cands.obs, prob)
        rows[k] = idx
    return rows


def top1_counts(ctx, rows, lab):
    rel = lab["dist"] <= config.LABEL_MAX_DIST
    pred_arc = np.where(rows >= 0, ctx.cands.arc_id[np.maximum(rows, 0)], -1)
    return int((pred_arc[rel] == lab["arc_id"][rel]).sum()), int(rel.sum())


def route_metrics(ctx, trip, pieces, ref_route):
    """Coverage (set) and sequence (order-aware) route scores, see pathsnap.evaluate."""
    pred_seq = [a for p in pieces for a in p]
    length = arc_lengths_m(trip, ctx.proj, pred_seq + [int(a) for a in ref_route])
    return route_scores(pred_seq, [int(a) for a in ref_route], length)


def tune(model: str, n_trips: int) -> HMMParams:
    preds = pd.read_parquet(config.CACHE_DIR / "preds" / f"{model}_oof.parquet").groupby("trip_id", sort=False)
    trip_ids = list(preds.groups)
    rng = np.random.default_rng(config.SEED)
    trip_ids = sorted(rng.choice(trip_ids, size=min(n_trips, len(trip_ids)), replace=False).tolist())
    inputs = [trip_inputs("train", t, preds) for t in trip_ids]
    grid = list(itertools.product([5.0, 10.0, 20.0, 40.0], [0.5, 1.0, 2.0]))
    rows = []
    base_hit = sum(top1_counts(ctx, argmax_rows(ctx, prob), lab)[0] for _, ctx, prob, lab in inputs)
    total = sum(top1_counts(ctx, argmax_rows(ctx, prob), lab)[1] for _, ctx, prob, lab in inputs)
    graphs = {}
    for beta, w in grid:
        params = HMMParams(beta=beta, emission_weight=w)
        t0 = time.perf_counter()
        hit = 0
        for trip, ctx, prob, lab in inputs:
            rows_hmm, g = viterbi(ctx, prob, params, graphs.get(trip.trip_id))
            graphs[trip.trip_id] = g
            hit += top1_counts(ctx, rows_hmm, lab)[0]
        rows.append({"beta": beta, "emission_weight": w, "train_top1_hmm": hit / total,
                     "train_top1_classifier": base_hit / total, "seconds": time.perf_counter() - t0})
        print(f"  beta {beta:5.1f}  w_e {w:3.1f}  top1 {hit / total:.4f}  (classifier {base_hit / total:.4f})"
              f"  {rows[-1]['seconds']:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "hmm"
    out.mkdir(parents=True, exist_ok=True)
    df.assign(model=model, tune_trips=json.dumps(trip_ids)).to_csv(out / f"tuning_{model}.csv", index=False)
    best = df.sort_values("train_top1_hmm", ascending=False).iloc[0]
    print(f"best: beta {best.beta}  emission_weight {best.emission_weight}  top1 {best.train_top1_hmm:.4f}")
    return HMMParams(beta=float(best.beta), emission_weight=float(best.emission_weight))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--baseline", default="distance_gaussian")
    ap.add_argument("--tune-trips", type=int, default=30)
    args = ap.parse_args()

    print(f"tuning decoder on train OOF predictions of {args.model} ...", flush=True)
    params = tune(args.model, args.tune_trips)
    base_params = HMMParams(beta=params.beta, emission_weight=1.0)

    preds = pd.read_parquet(config.CACHE_DIR / "preds" / f"{args.model}_test.parquet").groupby("trip_id", sort=False)
    bpreds = pd.read_parquet(config.CACHE_DIR / "preds" / f"{args.baseline}_test.parquet").groupby("trip_id", sort=False)
    per_trip = []
    for trip_id in preds.groups:
        trip, ctx, prob, lab = trip_inputs("student_test", trip_id, preds)
        bprob = bpreds.get_group(trip_id)["prob"].to_numpy()
        ref_route = load_route(config.DATA_ROOT / "student_test" / trip_id)
        t0 = time.perf_counter()
        rows_hmm, graph = viterbi(ctx, prob, params)
        t_hmm = time.perf_counter() - t0
        variants = {
            "classifier_only": argmax_rows(ctx, prob),
            "classifier_hmm": rows_hmm,
            "distance_hmm": viterbi(ctx, bprob, base_params, graph)[0],
            "distance_only": argmax_rows(ctx, bprob),
        }
        seg = segment_ids(ctx, params)
        for name, rows in variants.items():
            hit, n = top1_counts(ctx, rows, lab)
            cont, n_tr = plausible_continuity(ctx, graph, rows, seg)
            r = {"trip_id": trip_id, "variant": name, "top1_hits": hit, "n_reliable": n, "top1": hit / max(n, 1),
                 "continuity": cont, "n_transitions": n_tr}
            r.update(route_metrics(ctx, trip, build_route(ctx, graph, rows, seg), ref_route))
            if name == "classifier_hmm":
                r["decode_s"] = t_hmm
            per_trip.append(r)
        msg = "  ".join(f"{k}: {per_trip[-len(variants) + i]['top1']:.4f}" for i, k in enumerate(variants))
        print(f"{trip_id}  {msg}  (hmm {t_hmm:.1f}s)", flush=True)

    df = pd.DataFrame(per_trip)
    out = config.RESULTS_DIR / "hmm"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"per_trip_{args.model}.csv", index=False)
    def pooled_continuity(g):
        n = g["n_transitions"].sum()
        return float((g["continuity"].fillna(0) * g["n_transitions"]).sum() / n) if n else None

    summ = df.groupby("variant").apply(lambda g: pd.Series({
        "top1": g["top1_hits"].sum() / g["n_reliable"].sum(),
        "continuity_pooled": pooled_continuity(g),
        "coverage_f1": g["coverage_f1"].mean(), "sequence_f1": g["sequence_f1"].mean(),
        "sequence_precision": g["sequence_precision"].mean(), "sequence_recall": g["sequence_recall"].mean(),
        "route_mismatch": g["route_mismatch"].mean()}), include_groups=False)
    summ["model"] = args.model
    summ["params"] = json.dumps(params.as_dict())
    summ.to_csv(out / f"summary_{args.model}.csv")
    print("\n" + summ.drop(columns=["params"]).round(4).to_string())


if __name__ == "__main__":
    main()
