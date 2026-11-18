# PathSnap

PathSnap maps each GPS observation to a directed road arc. The supplied Student folder.zip has 80 train trips and 10 student_test trips. Each trip contains headerless whitespace-separated .track, .nodes, .arcs, and .route files:

- .track: longitude, latitude, timestamp in seconds
- .nodes: longitude, latitude; row number is the node ID
- .arcs: from-node, to-node; row number is the directed arc ID
- .route: chronological arc IDs, not point-aligned labels

The archive contains no evaluator-test partition. No evaluator labels have been used.

## Current data and model decisions

- Coordinates are projected to the WGS84 / UTM zone chosen per trip before distance calculations.
- A training-only candidate sweep selected a 250 m search radius and a 128-arc cap. This retains a mean 72.69 candidates per observation and has a 0.7298% training candidate miss rate. Trips 66, 02, and 54 have the largest per-trip miss rates.
- Point-level labels are an explicit monotone nearest-route-arc assignment from .route. The route sequence is not ground truth at every GPS point; trips 54 and 33 show strong evidence that this assumption can misalign labels. Their labels have not been changed.
- The feature cache contains 15 numeric features, including projected distance, rank, heading difference, gap to the nearest arc, projected movement on an arc, and connection to the eight nearest candidates at the previous point.
- The nearest-arc baseline is 48.65% pooled Top-1 over training observations, 50.71% mean fold Top-1, and 52.53% on the 10 student-comparison trips.
- A trip-grouped Random Forest + HMM run has completed. Its results, calibration check, fold logs, and per-trip comparisons are under experiments/hmm/.

## Python environment

The original pathsnap Conda environment is missing the ML/GIS packages. The existing cascade environment has scikit-learn and CUDA-enabled PyTorch; the machine reports an RTX 5060. Use its Python directly in PowerShell:

~~~powershell
$py = 'C:\Users\bhuvi\miniforge3\envs\cascade\python.exe'
$env:PYTHONPATH = "$PWD\src"
~~~

The agent's execution environment cannot resolve PyPI, so the XGBoost and Folium commands below must be run from a terminal that has package-network access.

## Reproduce the current CPU results

From this project folder:

~~~powershell
$py = 'C:\Users\bhuvi\miniforge3\envs\cascade\python.exe'
$env:PYTHONPATH = "$PWD\src"
& $py -m pathsnap.cli build-features --archive 'C:\Projects\ML Hackathon\Student folder.zip' --selection logs/candidate_selection_train.json --cache-dir data/interim/features
& $py -m pathsnap.cli baseline --cache-dir data/interim/features --experiment-dir experiments/model_comparison --folds 5
& $py -m pathsnap.cli hmm --cache-dir data/interim/features --experiment-dir experiments/hmm --folds 5 --seed 42
& $py -m pathsnap.cli summarize-hmm --cache-dir data/interim/features --experiment-dir experiments/hmm --folds 5
~~~

The HMM command reuses completed out-of-fold probability files and runs the missing folds. It fits on training trips only. The student partition is used only for a final development-stage comparison.

## Trip diagnostics and maps

After installing Folium in the selected environment, build two interactive overlays without changing any labels:

~~~powershell
& $py -m pip install 'folium>=0.19'
& $py -m pathsnap.cli diagnose-trips --archive 'C:\Projects\ML Hackathon\Student folder.zip' --cache-dir data/interim/features --output-dir reports/trip_diagnostics
~~~

The map layers show the GPS trace, assigned reference-route arcs, nearest-arc predictions, sampled baseline errors, and candidate misses. Outputs are reports/trip_diagnostics/trip_00000054_label_vs_nearest.html and trip_00000033_label_vs_nearest.html.

## XGBoost on the RTX 5060

Install the Windows XGBoost package in the CUDA-capable environment, run the small device check, then start grouped five-fold training:

~~~powershell
& $py -m pip install 'xgboost==3.4.2'
& $py -m pathsnap.cli xgb-smoke --out experiments/xgboost_gpu/gpu_smoke.json
if ($LASTEXITCODE -ne 0) { throw 'XGBoost CUDA smoke test failed; grouped training was not started.' }
& $py -m pathsnap.cli xgb-gpu --cache-dir data/interim/features --experiment-dir experiments/xgboost_gpu --folds 5 --seed 42
~~~

In another PowerShell window, nvidia-smi -l 1 can show GPU utilization while a fit is running. The model uses tree_method="hist" and device="cuda". The smoke test also checks that the installed XGBoost build reports CUDA support.

## Tests and outputs

Run the current unit tests with:

~~~powershell
$env:PYTHONPATH = "$PWD\src"
& $py -m unittest discover -s tests -v
~~~

The test suite covers metric projection, feature/label separation, grouped partitions, Viterbi transitions, empty-candidate and gap resets, and candidate context features.

Key generated files:

- reports/dataset_profile.json, reports/trip_inventory.csv
- logs/candidate_selection_train.json, logs/candidate_sweep_train.csv
- data/interim/features/ and per-split feature-build logs
- experiments/model_comparison/nearest_arc_baseline.json
- experiments/model_comparison/reverse_direction_diagnostic.json
- reports/trip_diagnostics/trip_label_diagnostics.json
- experiments/hmm/classifier_vs_hmm.json, experiments/hmm/hmm_trip_metrics.csv, and experiments/hmm/cv/fold_*.json

All reported results are computed from the supplied train and student-comparison partitions. No evaluator-test scores are available.
