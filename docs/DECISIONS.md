# Decisions log

Each entry: what we decided, why, and where the evidence lives. Dates are 2026-11-18.

| # | Decision | Why | Evidence |
|---|---|---|---|
| 1 | Work in metres in a per-trip UTM projection (zone of the track's median point). | Trips span several continents; degree distances are not metric. UTM is conformal with < 0.1 % scale error inside a zone. | `src/pathsnap/geo.py` |
| 2 | Derive one reference arc per GPS point by **monotonic DP alignment** of the track to the route (capped distance + heading penalty). | `.route` is only an ordered list of arcs; there is no per-point label. Monotonic order handles loops and repeated arcs (10 routes, up to 255 repeats); the heading term separates the two directions of a road (92 % of arcs have a reverse twin). | `src/pathsnap/labels.py`, `results/labels/labels_summary.csv` |
| 3 | Points more than **50 m** from their reference arc are unlabelled; trips with a median label distance above 50 m are excluded (train 33, train 54). | Typical label distance is 4–7 m (p99 on student-test 28 m). Trip 33's GPS is on roads but its route is ~120 m away (reference error); trip 54's whole track sits ~117 m off every road (coordinate offset). Both rules were set from train data. | `results/labels/labels_summary.csv` |
| 4 | Candidates: directed arcs within **50 m**, at most **16** per point. | Radius/K sweep on train: miss rate 0.11 % at ~7.8 candidates per point; a larger radius adds no reliable labels. | `results/candidates/sweep.csv` |
| 5 | The nearest-arc rule is the baseline to beat. | It is right for only 55 % of points but the reference is within the nearest two for 91 %. The gap is direction (twin arcs), so heading features matter most. | `results/candidates/ref_rank.csv` |
| 6 | Cross-validation is grouped by **trip** (5 folds, seed 352). | Points of one trip are strongly correlated; splitting by point would leak. | `scripts/04_compare_models.py` |
| 7 | Primary selection metric: **observation-level Top-1** on reliably labelled points (misses count as errors). Macro F1 and log loss are reported at pair level. | The decoder uses the ranking of candidates, not a 0.5 threshold. Log loss matters because probabilities become HMM emissions. | `src/pathsnap/evaluate.py` |
| 8 | Route continuity = share of consecutive predicted-arc changes that are **physically reachable** over the network (≤ max(3 × GPS distance, 100 m)). | Strict head-to-tail adjacency penalises correct predictions: at 1 Hz, consecutive points often skip short arcs (only 91 % of reference arcs get a point at all). | `src/pathsnap/hmm.py` |
| 9 | HMM decoder parameters are tuned on **train out-of-fold** predictions only. | The decoder must not see probabilities the classifier was trained on, and student-test stays a clean comparison set. | `scripts/05_hmm.py`, `results/hmm/` |
| 10 | Add a **long-range heading** (points ≥ 40 m apart, within ±180 s) next to the short one (feature set v2: 29 features; v1 had 26). | Error analysis: accuracy was 51 % below 1 m/s and 61 % at 1–3 m/s, mostly wrong-direction (twin) errors, because a stopped car has no usable short-range heading. With v2, LightGBM student-test Top-1 went 0.889 → 0.930, log loss 0.062 → 0.048. All models were re-run on v2 for the comparison. | `scripts/08_error_analysis.py`, `results/errors/`, `results/models/model_comparison_featureset_v1.csv` |
| 11 | Use the classifier's probabilities as HMM emissions (β = 10 m, emission weight 1, tuned on train OOF). | Student-test, v1 features: Top-1 0.890 → 0.921 with the HMM, plausible continuity 0.871 → 0.992, route F1 0.905 → 0.983. The classic distance-only HMM reaches 0.908 / 0.980, so the learned emission helps. Trip 40 is the exception: there the classic HMM is slightly better. | `results/hmm/summary_lightgbm.csv`, `results/hmm/per_trip_lightgbm.csv` |

| 12 | A label-free **GPS-offset correction** was built and then **disabled**; only its diagnostic (median distance to the nearest road, `gps_looks_offset`) is reported. | Only one supplied trip is offset (train 54: 84 m; every other trip ≤ 7.3 m). On it, the grid-searched shift fitted the wrong streets of a dense grid (median misfit 3.5 m), and route F1 fell from 0.947 to 0.133. Without correction the HMM already recovers the route (F1 0.947). It never triggers on normal trips. | `scripts/11_offset_check.py`, `results/pipeline/offset_check.csv` |
| 13 | The pipeline is hardened for unseen evaluator trips: flat or sub-folder layout, unsorted timestamps, missing `.route` (predict only), no road within reach, single-point trips; non-finite metrics become `null`. | The evaluator's ten trips are unseen, and a crash during the demo costs more than any accuracy point. | `scripts/09_robustness_check.py` |

## AI assistance

Code scaffolding and parts of the analysis were produced with an AI coding assistant (Claude).
Every experiment was run by the team, and every number comes from files in `results/`.
