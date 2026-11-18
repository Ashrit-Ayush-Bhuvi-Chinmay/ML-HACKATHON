# PathSnap — map matching of noisy GPS trajectories

UE24CS352A Machine Learning hackathon, PES University RR Campus.

Given a noisy GPS track and the road network around it, PathSnap recovers the sequence of road
segments the vehicle actually drove: a LightGBM classifier scores candidate road segments for every
GPS point, and an HMM with Viterbi decoding turns those scores into one connected route.

| Part | Folder | What |
|---|---|---|
| ML pipeline | `src/pathsnap/`, `scripts/` | data loading, label derivation, candidates, features, 11 models, Bayesian tuning, HMM/Viterbi |
| Results | `results/`, `artifacts/` | every experiment's numbers; the deployed model and every model we tried |
| API | `backend/` | FastAPI server (laptop 1) — see `backend/README.md` |
| UI | `ui/` | React + Vite evaluator interface (laptop 2) |
| Docs | `docs/` | `API_CONTRACT.md`, `DECISIONS.md` (every design decision + evidence), `FREEZE.md` (model hashes) |

## Results

| | Top-1 (points on the right segment) | Sequence F1 (route, in order) | Route mismatch |
|---|---|---|---|
| Student-test (10 trips) | 0.947 | 0.985 | 0.026 |
| Evaluator test (10 trips, one recorded pass) | **0.958** | **0.983** | **0.022** |

Distance-only baseline on the evaluator trips: Top-1 0.483, sequence F1 0.452.
Full comparison of 11 models: `results/models/model_comparison.csv`.

## Setup (Windows 10/11)

Needs **Python 3.12** (laptop 1) and **Node.js ≥ 22.12** (laptop 2). The course data folder
(`Provided Material/...`) is not in this repository.

```powershell
git clone <repo> pathsnap; cd pathsnap
.\setup.ps1 -Role api      # laptop 1: .venv with the ML package and the backend
.\setup.ps1 -Role ui       # laptop 2: UI packages (+ build check)
```

If PowerShell refuses to run scripts: `Set-ExecutionPolicy -Scope Process Bypass` in that window.

## Running the demo

**Laptop 1 (API)**
```powershell
.\run_api.ps1 -EvalDir "<folder with the 10 trip sub-folders>"
```
Prints the address to use (e.g. `http://192.168.1.23:8000`) and logs every request — keep this
window visible. Allow the port once, in an admin PowerShell:
`New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`

**Laptop 2 (UI)**
```powershell
.\run_ui.ps1
```
Open `http://localhost:5173`, type laptop 1's address into **API address**, pick a trip.

If the laptops cannot reach each other on college Wi-Fi (device-to-device traffic is often
blocked), put both on a phone hotspot. Both on one machine: run both scripts and use
`http://localhost:8000`.

## Reproducing the experiments

With the course data next to the repository (`../Provided Material/Student folder/`) or
`PATHSNAP_DATA_ROOT` set, run in order: `scripts/01_build_labels.py`, `02_candidate_sweep.py`,
`03_build_dataset.py`, `04_compare_models.py`, `07_tune.py`, `05_hmm.py`, `06_export.py`,
`12_route_metrics_from_preds.py`. Seeds are fixed (352). See `docs/DECISIONS.md` for why each
step is the way it is.

## AI assistance

Parts of the code and analysis were produced with AI coding assistants; every experiment was run
by the team and every reported number comes from files in `results/`.
