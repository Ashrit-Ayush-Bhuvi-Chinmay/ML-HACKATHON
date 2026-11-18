# Decoder experiments

This experiment reuses the cached candidate rows and classifier predictions from the main PathSnap runs. The search compares 45 configurations over five emission models, an equal-weight blend of their normalized scores, several Viterbi distance penalties, and a learned transition score. The raw inputs come from the supplied training and student-test splits.

The learned transition model is LightGBM. It uses arc geometry, direction, network distance, observed displacement and timing, candidate projection positions, and estimated speed. For each eligible pair of observations, the aligned reference arc pair supplies the positive example. Up to eight reachable alternatives supply the negative examples. Their class weights sum to the same weight as the positive example. During decoding, a clipped model logit is added to the existing distance-based transition score.

The fixed seed splits the 78 retained training trips into 30 to fit the transition model, 24 to select configurations, and 24 to check them. The Viterbi search uses saved out-of-fold emission scores for training trips. The shortlist contains the frozen setting, three previously reported reference settings, the training-selection winners for point Top-1 and mean sequence F1, and the highest-sequence-F1 learned-transition setting. The shortlist is then evaluated on the 24 check trips and all ten student-test trips. The existing emission models and some reference settings were developed before this experiment, so these are development comparisons.

Run from `ML_HACKATHON/branches/research` with the existing hackathon environment and prediction caches.

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
$env:PATHSNAP_CACHE_DIR = (Resolve-Path ../../pathsnap/cache).Path
$runDir = 'experiments/unrestricted_research/decoder_search'
..\..\.venv\Scripts\python.exe -B "$runDir/run.py" --output "$runDir/repeat"
..\..\.venv\Scripts\python.exe -B "$runDir/ablate.py" --results "$runDir/repeat"
```

The output directory must be new. `summary.csv` contains point-weighted Top-1 and trip-averaged coverage and sequence F1. `selection_per_trip.csv` contains the full search and `comparison_per_trip.csv` contains the shortlisted runs. `configs.csv`, `train_split.csv`, and `transition_model.txt` record the settings, trip assignments, and fitted transition model. Each evaluated trip also checks that the experimental decoder reproduces the production Viterbi output for the frozen setting.

`ablate.py` reads the chosen results directory and compares the selected blend with its learned transition term turned off. It writes `ablation_summary.csv` and `ablation_per_trip.csv`.

| Decoder | Training check Top-1 | Training check sequence F1 | Student-test Top-1 | Student-test sequence F1 |
|---|---:|---:|---:|---:|
| Frozen tuned LightGBM | 95.43% | 97.00% | 94.72% | 98.50% |
| Blend with distance transitions | 95.12% | 97.27% | 94.71% | 98.77% |
| Blend with learned transitions | 95.23% | 97.86% | 94.59% | 99.06% |

Learning the transition score improved mean sequence F1 over the matched blend by 0.59 percentage points on the training check and 0.29 points on student-test. Compared with the frozen system, the learned blend improved sequence F1 on six student-test trips, tied on three, and lost on one. Its point Top-1 was lower by 25 correct assignments out of 19,133 scored points.
