"""Step 3 driver: how large must the candidate set be?

For each trip, generate every arc within 200 m of each GPS point, then measure for a grid of
(radius R, max candidates K):
    miss rate     share of reliably labelled points whose reference arc is not a candidate
    mean set size candidates per point (cost of every later stage)
    empty rate    share of points with no candidate at all
Also records the distance rank of the reference arc (0 = nearest), i.e. how often the plain
"nearest arc" rule would already be right.

Outputs: results/candidates/sweep.csv, results/candidates/ref_rank.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.candidates import generate
from pathsnap.geo import Projector
from pathsnap.io import list_trips, load_trip
from pathsnap.network import build_network

RADII = [25, 50, 75, 100, 150, 200]
KS = [4, 8, 12, 16, 24, 32, None]


def main() -> None:
    bad = config.bad_reference_trips()
    acc = {}       # (split, R, K) -> [n_rel, n_hit, sum_size, n_empty, n_points]
    rank_rows = []
    for split in config.SPLITS:
        for trip_dir in list_trips(config.DATA_ROOT / split):
            if (split, trip_dir.name) in bad:
                print(f"skip {split}/{trip_dir.name} (unusable reference)")
                continue
            t0 = time.perf_counter()
            trip = load_trip(trip_dir)
            lab = np.load(config.CACHE_DIR / "labels" / split / f"{trip.trip_id}.npz")
            proj = Projector.for_track(trip.track)
            px, py = proj.to_xy(trip.track[:, 0], trip.track[:, 1])
            net = build_network(trip, proj, px, py)
            t_net = time.perf_counter() - t0
            c = generate(net, px, py, radius=max(RADII), k_max=None)
            t_cand = time.perf_counter() - t0 - t_net

            n = trip.n_points
            reliable = lab["dist"] <= config.LABEL_MAX_DIST
            is_ref = c.arc_id == lab["arc_id"][c.obs]
            for R in RADII:
                for K in KS:
                    m = c.dist <= R
                    if K is not None:
                        m &= c.rank < K
                    hit = np.bincount(c.obs[m & is_ref], minlength=n) > 0
                    size = np.bincount(c.obs[m], minlength=n)
                    a = acc.setdefault((split, R, K), [0, 0, 0, 0, 0])
                    a[0] += int(reliable.sum())
                    a[1] += int((hit & reliable).sum())
                    a[2] += int(size.sum())
                    a[3] += int((size == 0).sum())
                    a[4] += n
            # rank of the reference arc among all candidates within 200 m
            ref_rank = np.full(n, -1)
            ref_rank[c.obs[is_ref]] = c.rank[is_ref]
            rr = ref_rank[reliable]
            rank_rows.append({"split": split, "trip_id": trip.trip_id, "n_reliable": int(reliable.sum()),
                              "rank0": float(np.mean(rr == 0)), "rank_le1": float(np.mean((rr >= 0) & (rr <= 1))),
                              "rank_le3": float(np.mean((rr >= 0) & (rr <= 3))), "missing_200m": float(np.mean(rr < 0)),
                              "n_arcs_full": net.n_arcs_full, "n_arcs_corridor": len(net.arc_ids),
                              "t_network_s": t_net, "t_candidates_s": t_cand})
            print(f"{split:>12} {trip.trip_id} arcs {net.n_arcs_full:>7} -> {len(net.arc_ids):>6} corridor"
                  f"  net {t_net:4.1f}s  cand {t_cand:4.1f}s  ref nearest {np.mean(rr == 0):.2%}", flush=True)

    rows = []
    for (split, R, K), (n_rel, n_hit, s, e, n) in acc.items():
        rows.append({"split": split, "radius_m": R, "k_max": K if K is not None else "all",
                     "miss_rate": 1 - n_hit / n_rel, "mean_set_size": s / n, "empty_rate": e / n,
                     "n_points": n, "n_reliable": n_rel})
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep.csv", index=False)
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(out / "ref_rank.csv", index=False)

    for split in config.SPLITS:
        d = df[df.split == split]
        print(f"\n=== {split}: miss rate (%) by radius (rows) and K (columns) ===")
        print((d.pivot(index="radius_m", columns="k_max", values="miss_rate") * 100).round(2).to_string())
        print(f"--- {split}: mean candidates per point ---")
        print(d.pivot(index="radius_m", columns="k_max", values="mean_set_size").round(1).to_string())
        w = rk[rk.split == split]
        ww = w["n_reliable"] / w["n_reliable"].sum()
        print(f"--- {split}: reference arc is the nearest arc for {np.sum(ww * w['rank0']):.2%} of points; "
              f"within nearest 2: {np.sum(ww * w['rank_le1']):.2%}; nearest 4: {np.sum(ww * w['rank_le3']):.2%}")


if __name__ == "__main__":
    main()
