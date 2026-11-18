"""Research R3: two-stage (stacked) candidate scoring.

Stage 1 = the submitted tuned LightGBM. Its out-of-fold (OOF) probabilities on train and its
student-test probabilities already exist (cache/preds), so no stage-1 retraining is needed.
Stage 2 adds, for every (point i, candidate arc a), what the NEIGHBOURING points think:
    p_same_{k}   stage-1 probability of the same arc a at point i+k   (k = -5,-2,-1,+1,+2,+5; 0 if absent)
    best_is_a_{k} / best_touches_a_{k}  neighbour's top arc is a / shares a node with a
    p_self, rank_self, gap_to_best      this candidate's own stage-1 standing at point i
Leakage control: stage-2 training rows carry OOF stage-1 probabilities, and stage-2 CV uses the
SAME 5 trip-grouped folds as stage 1, so a validation trip never influences its own features.
Then the R1 decoder (emission weight re-checked, since stage-2 already uses neighbour context)
on the same 20 train trips as R1, and student-test once. Evaluator set NOT used.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.evaluate import point_keys, top1_accuracy
from pathsnap.features import FEATURES
from pathsnap.io import load_trip
from pathsnap.models import SPECS

spec = importlib.util.spec_from_file_location("r1", Path(__file__).with_name("r1_decoder_research.py"))
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)

OUT = config.RESULTS_DIR / "research"
OFFSETS = (-5, -2, -1, 1, 2, 5)


def stage2_features(P: pd.DataFrame, prob: np.ndarray, split: str) -> pd.DataFrame:
    """P: candidate table (trip_id, obs, arc_id, ...) in candidate order; prob: stage-1 scores."""
    df = P[["trip_id", "obs", "arc_id"]].copy()
    df["p"] = prob
    g = df.groupby(["trip_id", "obs"], sort=False)["p"]
    df["p_self"] = prob
    df["gap_to_best"] = g.transform("max") - prob
    df["rank_self"] = g.rank(ascending=False, method="first")
    # neighbour's best arc per point
    best = df.loc[g.idxmax(), ["trip_id", "obs", "arc_id"]].rename(columns={"arc_id": "best_arc"})
    nodes = {}
    for tid in df["trip_id"].unique():
        nodes[tid] = load_trip(config.DATA_ROOT / split / tid).arcs
    for k in OFFSETS:
        sh = df[["trip_id", "obs", "arc_id", "p"]].rename(columns={"p": f"p_same_{k:+d}"})
        sh["obs"] = sh["obs"] - k  # value at point i+k is attached to point i
        df = df.merge(sh, on=["trip_id", "obs", "arc_id"], how="left")
        bk = best.copy()
        bk["obs"] = bk["obs"] - k
        df = df.merge(bk.rename(columns={"best_arc": f"best_{k:+d}"}), on=["trip_id", "obs"], how="left")
        df[f"p_same_{k:+d}"] = df[f"p_same_{k:+d}"].fillna(0.0)
        b = df[f"best_{k:+d}"].fillna(-1).astype(np.int64).to_numpy()
        a = df["arc_id"].to_numpy()
        touch = np.zeros(len(df), dtype=np.float32)
        for tid, idx in df.groupby("trip_id", sort=False).indices.items():
            arcs = nodes[tid]
            ok = b[idx] >= 0
            bi = np.where(ok, b[idx], 0)
            au, av = arcs[a[idx], 0], arcs[a[idx], 1]
            bu, bv = arcs[bi, 0], arcs[bi, 1]
            touch[idx] = (ok & ((au == bu) | (au == bv) | (av == bu) | (av == bv))).astype(np.float32)
        df[f"best_is_a_{k:+d}"] = (b == a).astype(np.float32)
        df[f"best_touches_a_{k:+d}"] = touch
        df = df.drop(columns=[f"best_{k:+d}"])
    assert len(df) == len(P)
    return df


S2_COLS = ["p_self", "gap_to_best", "rank_self"] + [f"{n}_{k:+d}" for k in OFFSETS
                                                       for n in ("p_same", "best_is_a", "best_touches_a")]


def main() -> None:
    t0 = time.perf_counter()
    P_tr = pd.read_parquet(config.CACHE_DIR / "dataset" / "train_pairs.parquet")
    P_te = pd.read_parquet(config.CACHE_DIR / "dataset" / "student_test_pairs.parquet")
    oof = pd.read_parquet(config.CACHE_DIR / "preds" / "lightgbm_tuned_oof.parquet")
    tst = pd.read_parquet(config.CACHE_DIR / "preds" / "lightgbm_tuned_test.parquet")
    assert np.array_equal(oof["obs"].to_numpy(), P_tr["obs"].to_numpy())
    assert np.array_equal(tst["obs"].to_numpy(), P_te["obs"].to_numpy())

    F_tr = stage2_features(P_tr, oof["prob"].to_numpy(), "train")
    F_te = stage2_features(P_te, tst["prob"].to_numpy(), "student_test")
    print(f"stage-2 features built ({time.perf_counter() - t0:.0f}s)", flush=True)
    cols = FEATURES + S2_COLS
    X_tr = np.hstack([P_tr[FEATURES].to_numpy(), F_tr[S2_COLS].to_numpy()])
    X_te = np.hstack([P_te[FEATURES].to_numpy(), F_te[S2_COLS].to_numpy()])
    y, rel = P_tr["y"].to_numpy(), P_tr["reliable"].to_numpy()

    # same 5 folds as the comparison script (stage 1)
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=config.SEED).split(P_tr, groups=P_tr["trip_id"]))
    params = dict(SPECS["lightgbm"].params)  # untuned defaults: fast, and keeps the stage-2 effect clean
    oof2 = np.zeros(len(P_tr))
    for tr, va in folds:
        tr = tr[rel[tr]]
        m = SPECS["lightgbm"].build(params, config.SEED).fit(X_tr[tr], y[tr])
        oof2[va] = m.predict_proba(X_tr[va])[:, 1]
    print(f"stage-2 OOF done ({time.perf_counter() - t0:.0f}s)", flush=True)
    m_full = SPECS["lightgbm"].build(params, config.SEED).fit(X_tr[rel], y[rel])
    p2_te = m_full.predict_proba(X_te)[:, 1]
    imp = pd.Series(m_full.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)

    # classifier-only top-1 (OOF, all train trips) stage 1 vs stage 2
    codes = {t: i for i, t in enumerate(sorted(P_tr["trip_id"].unique()))}
    keys = point_keys(P_tr["trip_id"].map(codes).to_numpy(), P_tr["obs"].to_numpy())
    Q = pd.read_parquet(config.CACHE_DIR / "dataset" / "train_points.parquet")
    n_rel = int(Q.loc[Q["trip_id"].isin(codes), "reliable"].sum())
    res = {"train_oof_classifier_top1_stage1": top1_accuracy(keys[rel], y[rel], oof["prob"].to_numpy()[rel], n_rel),
           "train_oof_classifier_top1_stage2": top1_accuracy(keys[rel], y[rel], oof2[rel], n_rel)}
    print(res, flush=True)

    # decoding on the same 20 train trips as R1 (same sampling), then student-test once
    rng = np.random.default_rng(config.SEED)
    trips = sorted(rng.choice(oof["trip_id"].unique(), size=20, replace=False).tolist())
    train_b = {t: r1.trip_bundle("train", t) for t in trips}
    test_b = {t: r1.trip_bundle("student_test", t) for t in sorted(P_te["trip_id"].unique())}

    def probs(bundles, P, prob):
        g = P[["trip_id", "obs", "arc_id"]].assign(prob=prob).groupby("trip_id", sort=False)
        return {t: r1.probs_for(b, g, t) for t, b in bundles.items()}

    best = r1.HMMParams(beta=10.0, emission_weight=0.5, unreachable_logp=-25.0, backward_tol=30.0, gap_dt=300.0)
    res["train_hmm_stage1_R1"] = r1.run(train_b, probs(train_b, P_tr, oof["prob"].to_numpy()), best)
    s2_train = probs(train_b, P_tr, oof2)
    sweep = {}
    for w in (0.2, 0.3, 0.5, 0.7, 1.0):
        sweep[w] = r1.run(train_b, s2_train, r1.replace(best, emission_weight=w))
        print(f"stage-2 + HMM, emission weight {w}: {sweep[w]}", flush=True)
    res["train_hmm_stage2_by_weight"] = {str(k): v for k, v in sweep.items()}
    w_best = max(sweep, key=lambda w: (round(sweep[w]["top1"], 4), -sweep[w]["route_mismatch"]))
    res["selected_emission_weight"] = w_best

    res["test_hmm_stage1_R1"] = r1.run(test_b, probs(test_b, P_te, tst["prob"].to_numpy()), best)
    res["test_hmm_stage2"] = r1.run(test_b, probs(test_b, P_te, p2_te), r1.replace(best, emission_weight=w_best))
    res["stage2_top_features"] = imp.head(12).round(0).to_dict()
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r3_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("test") or k.startswith("selected")}, indent=2,
                     default=float))


if __name__ == "__main__":
    main()
