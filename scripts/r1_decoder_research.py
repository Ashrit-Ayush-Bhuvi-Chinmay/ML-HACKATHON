"""Post-submission research: can Top-1 / route mismatch improve without retraining?

All choices are made on TRAIN out-of-fold predictions (a fixed sample of train trips); the
student-test split is used once at the end to confirm. The evaluator set is NOT used here.

  A. error anatomy of the frozen system on student-test: how many remaining errors are
     junction-boundary ambiguity (predicted arc shares a node with the reference arc and the GPS
     point lies within 10 m of that node) - i.e. how much is realistically recoverable
  B. random search over ALL decoder settings (beta, emission weight, unreachable penalty,
     backward tolerance, gap threshold) on train OOF
  C. emission source: tuned LightGBM vs averaged ensembles (+XGBoost, +listwise ranker)
  D. decoding: Viterbi (best path) vs posterior / forward-backward (best segment per point)

Run with PYTHONPATH=<research>/src and PATHSNAP_CACHE_DIR=<pathsnap>/cache (read-only use).
Output: results/research/*.csv, results/research/summary.json
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pathsnap import config
from pathsnap.evaluate import arc_lengths_m, best_per_point, route_scores
from pathsnap.features import prepare_trip
from pathsnap.hmm import (HMMParams, RoadGraph, _norm_log, build_route, plausible_continuity,
                          route_distance, segment_ids)
from pathsnap.io import load_route, load_trip

OUT = config.RESULTS_DIR / "research"
FROZEN = HMMParams(beta=20.0, emission_weight=1.0)  # the submitted decoder settings
N_TRAIN_TRIPS = 20


# ----------------------------------------------------------------------------- data
def load_probs(models: list[str], split: str) -> pd.DataFrame:
    suffix = "oof" if split == "train" else "test"
    dfs = [pd.read_parquet(config.CACHE_DIR / "preds" / f"{m}_{suffix}.parquet") for m in models]
    for d in dfs[1:]:
        assert np.array_equal(d["obs"].to_numpy(), dfs[0]["obs"].to_numpy())
        assert np.array_equal(d["arc_id"].to_numpy(), dfs[0]["arc_id"].to_numpy())
    out = dfs[0][["trip_id", "obs", "arc_id"]].copy()
    out["prob"] = np.mean([d["prob"].to_numpy() for d in dfs], axis=0)
    return out


def trip_bundle(split: str, trip_id: str):
    d = config.DATA_ROOT / split / trip_id
    trip = load_trip(d)
    ctx = prepare_trip(trip, config.CAND_RADIUS, config.CAND_K)
    lab = np.load(config.CACHE_DIR / "labels" / split / f"{trip_id}.npz")
    return {"trip": trip, "ctx": ctx, "lab": lab, "graph": RoadGraph(ctx, 1500.0),
            "ref": load_route(d).tolist()}


def probs_for(bundle, preds_by_trip, trip_id):
    g = preds_by_trip.get_group(trip_id)
    assert np.array_equal(g["obs"].to_numpy(), bundle["ctx"].cands.obs)
    return g["prob"].to_numpy()


# ----------------------------------------------------------------------------- lattice + decoders
def lattice(ctx, graph, prob, p: HMMParams):
    c = ctx.cands
    if len(c.obs) == 0:
        return []
    starts = np.flatnonzero(np.r_[True, c.obs[1:] != c.obs[:-1]])
    ends = np.r_[starts[1:], len(c.obs)]
    seg = segment_ids(ctx, p)
    chains, cur, prev = [], [], None
    for o, s, e in zip(c.obs[starts], starts, ends):
        rows = np.arange(s, e)
        emis = p.emission_weight * _norm_log(prob[s:e])
        if prev is not None and seg[o] != seg[prev[0]]:
            chains.append(cur)
            cur, prev = [], None
        if prev is None:
            cur.append((o, rows, emis, None))
        else:
            po, prows = prev
            d_gps = float(np.hypot(ctx.px[o] - ctx.px[po], ctx.py[o] - ctx.py[po]))
            d_route = route_distance(graph, ctx, c.local[prows], c.t[prows], c.local[rows], c.t[rows],
                                     p.backward_tol)
            logt = np.where(np.isfinite(d_route), -np.abs(d_route - d_gps) / p.beta, p.unreachable_logp)
            cur.append((o, rows, emis, logt))
        prev = (o, rows)
    chains.append(cur)
    return chains


def decode_viterbi(chains, n):
    best = np.full(n, -1, dtype=np.int64)
    for ch in chains:
        score, bps = None, []
        for o, rows, emis, logt in ch:
            if logt is None:
                score = emis.copy()
                bps.append(None)
            else:
                tot = score[:, None] + logt
                bp = np.argmax(tot, axis=0)
                score = tot[bp, np.arange(len(rows))] + emis
                bps.append(bp)
        j = int(np.argmax(score))
        for k in range(len(ch) - 1, -1, -1):
            best[ch[k][0]] = ch[k][1][j]
            if bps[k] is not None:
                j = int(bps[k][j])
    return best


def decode_posterior(chains, n):
    best = np.full(n, -1, dtype=np.int64)
    for ch in chains:
        alphas = []
        a = None
        for o, rows, emis, logt in ch:
            a = emis.copy() if logt is None else logsumexp(a[:, None] + logt, axis=0) + emis
            alphas.append(a)
        b = np.zeros(len(ch[-1][1]))
        for k in range(len(ch) - 1, -1, -1):
            o, rows, emis, logt = ch[k]
            post = alphas[k] + b
            best[o] = rows[int(np.argmax(post))]
            if logt is not None:
                b = logsumexp(logt + (emis + b)[None, :], axis=1)
    return best


def evaluate(bundle, rows, p: HMMParams):
    ctx, lab, trip, graph = bundle["ctx"], bundle["lab"], bundle["trip"], bundle["graph"]
    rel = lab["dist"] <= config.LABEL_MAX_DIST
    pred = np.where(rows >= 0, ctx.cands.arc_id[np.maximum(rows, 0)], -1)
    hits = int((pred[rel] == lab["arc_id"][rel]).sum())
    seg = segment_ids(ctx, p)
    cont, n_tr = plausible_continuity(ctx, graph, rows, seg)
    pieces = build_route(ctx, graph, rows, seg)
    seq = [a for pc in pieces for a in pc]
    sc = route_scores(seq, bundle["ref"], arc_lengths_m(trip, ctx.proj, seq + bundle["ref"]))
    return {"hits": hits, "n": int(rel.sum()), "cont_ok": (cont or 0) * n_tr, "n_tr": n_tr, **sc}


def run(bundles, probs, p, decoder="viterbi", per_trip=False):
    res = []
    for tid, b in bundles.items():
        ch = lattice(b["ctx"], b["graph"], probs[tid], p)
        rows = (decode_viterbi if decoder == "viterbi" else decode_posterior)(ch, len(b["ctx"].px))
        res.append({"trip_id": tid, **evaluate(b, rows, p)})
    df = pd.DataFrame(res)
    agg = {"top1": df["hits"].sum() / df["n"].sum(),
           "continuity": df["cont_ok"].sum() / max(df["n_tr"].sum(), 1),
           "sequence_f1": df["sequence_f1"].mean(), "coverage_f1": df["coverage_f1"].mean(),
           "route_mismatch": df["route_mismatch"].mean()}
    return (agg, df) if per_trip else agg


# ----------------------------------------------------------------------------- A: error anatomy
def error_anatomy(bundles, probs):
    rows_out = []
    for tid, b in bundles.items():
        ctx, lab, trip = b["ctx"], b["lab"], b["trip"]
        rows = decode_viterbi(lattice(ctx, b["graph"], probs[tid], FROZEN), len(ctx.px))
        rel = lab["dist"] <= config.LABEL_MAX_DIST
        pred = np.where(rows >= 0, ctx.cands.arc_id[np.maximum(rows, 0)], -1)
        ref = lab["arc_id"]
        for i in np.flatnonzero(rel & (pred != ref)):
            if pred[i] < 0:
                rows_out.append((tid, "unmatched", np.nan))
                continue
            pu, pv = trip.arcs[pred[i]]
            ru, rv = trip.arcs[ref[i]]
            if pu == rv and pv == ru:
                rows_out.append((tid, "twin", np.nan))
                continue
            shared = {pu, pv} & {ru, rv}
            if shared:
                node = shared.pop()
                nx, ny = ctx.proj.to_xy(trip.nodes[node, 0], trip.nodes[node, 1])
                d = float(np.hypot(ctx.px[i] - nx, ctx.py[i] - ny))
                rows_out.append((tid, "junction_boundary" if d <= 10 else "adjacent_far", d))
            else:
                rows_out.append((tid, "other_road", np.nan))
    df = pd.DataFrame(rows_out, columns=["trip_id", "kind", "dist_to_shared_node_m"])
    n_rel = sum(int((b["lab"]["dist"] <= config.LABEL_MAX_DIST).sum()) for b in bundles.values())
    return df, (df["kind"].value_counts() / n_rel).to_dict()


# ----------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    summary = {}

    oof = load_probs(["lightgbm_tuned"], "train")
    rng = np.random.default_rng(config.SEED)
    trips = sorted(rng.choice(oof["trip_id"].unique(), size=N_TRAIN_TRIPS, replace=False).tolist())
    print(f"building {len(trips)} train trips + 10 student-test trips ...", flush=True)
    train = {t: trip_bundle("train", t) for t in trips}
    test_ids = sorted(load_probs(["lightgbm_tuned"], "student_test")["trip_id"].unique())
    test = {t: trip_bundle("student_test", t) for t in test_ids}

    def probs(models, split, bundles):
        g = load_probs(models, split).groupby("trip_id", sort=False)
        return {t: probs_for(b, g, t) for t, b in bundles.items()}

    p_train = probs(["lightgbm_tuned"], "train", train)
    p_test = probs(["lightgbm_tuned"], "student_test", test)

    # A
    anat, frac = error_anatomy(test, p_test)
    anat.to_csv(OUT / "A_error_anatomy_student_test.csv", index=False)
    summary["A_error_share_of_points_student_test"] = {k: round(v, 4) for k, v in frac.items()}
    print("A error share of reliable points:", summary["A_error_share_of_points_student_test"], flush=True)

    # B
    base = run(train, p_train, FROZEN)
    print(f"B frozen on train: {base}", flush=True)
    trials = [("frozen", FROZEN, base)]
    for i in range(20):
        p = HMMParams(beta=float(rng.choice([5, 10, 15, 20, 30, 40])),
                      emission_weight=float(rng.choice([0.5, 0.7, 1.0, 1.4, 2.0])),
                      unreachable_logp=float(rng.choice([-15, -25, -40, -80])),
                      backward_tol=float(rng.choice([5, 15, 30, 50])),
                      gap_dt=float(rng.choice([30, 60, 120, 300])))
        r = run(train, p_train, p)
        trials.append((f"t{i}", p, r))
        print(f"B {i:2d} {p.beta:4.0f} {p.emission_weight:3.1f} {p.unreachable_logp:4.0f} {p.backward_tol:3.0f} "
              f"{p.gap_dt:4.0f} -> top1 {r['top1']:.4f} mismatch {r['route_mismatch']:.4f}", flush=True)
    tb = pd.DataFrame([{"trial": n, **p.as_dict(), **r} for n, p, r in trials])
    tb.to_csv(OUT / "B_decoder_search_train.csv", index=False)
    best_name, best_p, best_r = max(trials, key=lambda x: (round(x[2]["top1"], 4), -x[2]["route_mismatch"]))
    summary["B_best_decoder"] = {"trial": best_name, **best_p.as_dict(), **best_r, "frozen_train": base}

    # C
    sources = {"lgbm_tuned": ["lightgbm_tuned"], "lgbm+xgb": ["lightgbm_tuned", "xgboost"],
               "lgbm+xgb+ranker": ["lightgbm_tuned", "xgboost", "ranker_torch"]}
    c_rows = []
    for name, models in sources.items():
        r = run(train, probs(models, "train", train), best_p)
        c_rows.append({"source": name, **r})
        print(f"C {name:16s} top1 {r['top1']:.4f} mismatch {r['route_mismatch']:.4f}", flush=True)
    cdf = pd.DataFrame(c_rows)
    cdf.to_csv(OUT / "C_emission_source_train.csv", index=False)
    best_src = cdf.sort_values(["top1", "route_mismatch"], ascending=[False, True]).iloc[0]["source"]

    # D
    pb = probs(sources[best_src], "train", train)
    d_rows = []
    for dec in ("viterbi", "posterior"):
        r = run(train, pb, best_p, decoder=dec)
        d_rows.append({"decoder": dec, **r})
        print(f"D {dec:9s} top1 {r['top1']:.4f} cont {r['continuity']:.4f} mismatch {r['route_mismatch']:.4f}", flush=True)
    ddf = pd.DataFrame(d_rows)
    ddf.to_csv(OUT / "D_decoding_train.csv", index=False)
    best_dec = ddf.sort_values(["top1", "route_mismatch"], ascending=[False, True]).iloc[0]["decoder"]
    summary["selected_on_train"] = {"decoder_params": best_p.as_dict(), "emission_source": best_src,
                                    "decoding": best_dec}

    # confirmation on student-test (once): frozen system vs research selection
    frozen_test, frozen_pt = run(test, p_test, FROZEN, per_trip=True)
    sel_test, sel_pt = run(test, probs(sources[best_src], "student_test", test), best_p,
                           decoder=best_dec, per_trip=True)
    frozen_pt.assign(system="frozen").to_csv(OUT / "test_per_trip_frozen.csv", index=False)
    sel_pt.assign(system="research").to_csv(OUT / "test_per_trip_research.csv", index=False)
    summary["student_test"] = {"frozen": frozen_test, "research_selection": sel_test}
    summary["minutes"] = round((time.perf_counter() - t_start) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["student_test"], indent=2, default=float))
    print("selected:", summary["selected_on_train"])


if __name__ == "__main__":
    main()
