# Post-submission research log

Branch `research` (folder `branches/research`), started from `core-ml` after the submission was
frozen. Every experiment, its evidence and whether it was adopted is recorded here, in order.
Times are 2026-11-18, local; commit hashes make every entry verifiable.

## Ground rules

- Every choice is made on **train** (out-of-fold predictions, trip-grouped) and confirmed on
  **student-test**. The **evaluator** set is never used to train, tune or choose.
- The evaluator set is run at most once per *pre-declared* final version, and the result is kept
  whatever it is. Never "run several versions and keep the best".
- The submitted system stays frozen (`docs/FREEZE.md`, hash `af093e19…`).
- Route-level metrics (sequence / coverage F1, route mismatch) compare against `.route` directly and
  do not depend on per-point labels, so they referee any change to the labels.

## Baseline (submitted, frozen)

| When | Commit | Event |
|---|---|---|
| 12:06 | `c74bcec` / `1d5738a` | Final system frozen: Bayesian-tuned LightGBM + HMM (β 20 m, emission weight 1). Student-test Top-1 0.947, sequence F1 0.985, route mismatch 0.0255. |
| 13:29 | `5f9a964` | One recorded evaluator pass of the frozen system (hash verified): Top-1 0.958, sequence F1 0.983, coverage F1 0.989, continuity 0.993, mismatch 0.022. No changes afterwards. |

## Experiments

### R1 — decoder settings, emission ensembles, decoding method (13:30–13:42, commit `45d00b8`)

Script `scripts/r1_decoder_research.py`; results `results/research/` (`A_*`, `B_*`, `C_*`, `D_*`, `summary.json`).

- **A. Error anatomy** of the frozen system on student-test (share of reliable points):
  wrong direction (twin arc) 1.93 %, **junction boundary** (neighbouring arc, point ≤ 10 m from
  the shared node) 1.66 %, neighbouring arc further away 1.14 %, other road 0.56 %.
  About a third of the remaining error is label ambiguity at intersections, which caps Top-1 at
  roughly 97.5 % with the current labels.
- **B. Random search** (20 configurations + the frozen one) over β, emission weight, unreachable
  penalty, backward tolerance and gap threshold, on train OOF predictions of 20 train trips.
  Best: β 10, emission weight **0.5**, unreachable −25, backward tolerance 30 m, gap **300 s**.
  Train Top-1 0.946 → 0.954, mismatch 0.086 → 0.075. Lower emission weight = trust the road
  network more than each noisy point. The submitted search had the gap fixed at 60 s.
- **C. Emission source** with the B settings: LightGBM alone 0.9542, +XGBoost 0.9540,
  +XGBoost +ranker 0.9526. No gain from ensembling; the models make the same mistakes.
- **D. Decoding:** Viterbi 0.9542 vs posterior (forward-backward) 0.9534. Viterbi kept.
- **Student-test confirmation (once):** Top-1 0.9472 → **0.9517**, sequence F1 0.985 → 0.989,
  route mismatch 0.0255 → **0.0187**, continuity 0.995 → 0.998.
- Caveat: 21 random trials on 20 trips; part of the train gain is search luck. Student-test gain
  (+0.45 points) is the honest estimate.
- **Status: candidate.** Not in the submission. Decoder settings only; same trained model.

### R2 — labels from smoothed progress along the route (13:46–14:00)

Script `scripts/r2_progress_labels.py`. Hypothesis: nearest-arc labels flip between the two arcs at
an intersection because of GPS jitter; smoothing the vehicle's progress along the known route
(rolling median + isotonic regression) puts the arc boundary where the vehicle actually crossed.
Measured: labels changed, A→B→A flips, direction agreement at ≥ 5 m/s; then the same tuned
LightGBM retrained on new vs old labels, both decoded with the R1 decoder, compared on route-level
metrics (train OOF, 20 trips, then student-test once) and Top-1 under both label definitions.

Results (`results/research/r2_summary.json`, `r2_label_stats.csv`):

- Label quality: 1.44 % of points got a different arc; A→B→A flips 0 → 0 (the monotone DP never
  flipped back and forth, so the "jitter" hypothesis was wrong); direction agreement at ≥ 5 m/s
  0.9990 → 0.9989. The new method only moves where boundaries fall.
- Train (3-fold OOF, 20 trips decoded): sequence F1 0.9526 → 0.9529, mismatch 0.0758 → 0.0747,
  Top-1 judged by old labels 0.9555 → 0.9513.
- Student-test: sequence F1 0.9888 → 0.9892, mismatch 0.0187 → 0.0179 (both noise-level);
  Top-1 by old labels 0.9517 → 0.9515; Top-1 of the new model judged by **its own** labels 0.9600.
- **Verdict: not adopted.** Route-level quality (the referee) is unchanged; the 0.960 is the
  ruler changing, not the system improving. Conclusion for the report: the remaining point errors
  are not label noise — they are genuinely ambiguous positions near intersections and direction
  choices when the vehicle is stopped.

## Current best (research candidate)

R1 decoder settings with the frozen trained model: student-test Top-1 0.9517, sequence F1 0.9888,
route mismatch 0.0187. Not run on the evaluator set.
