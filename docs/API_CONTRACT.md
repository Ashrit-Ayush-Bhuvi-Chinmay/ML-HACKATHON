# PathSnap API contract (v0.1)

For the API and UI team. The model pipeline is being built in parallel; this contract lets
both sides work without waiting for each other.

## Tech stack

- **Backend: FastAPI** (required by the brief), run with uvicorn on **laptop 1**.
- **UI: your choice**, running on **laptop 2** and talking to the API over HTTP. Two easy options:
  - Streamlit + `streamlit-folium` (Python, quickest; the map can be a Folium map), or
  - a plain HTML page + Leaflet (nothing to install on laptop 2; enable CORS on the API).
- Python 3.12. Put dependencies in `api/requirements.txt` and `ui/requirements.txt`, so that
  laptop 2 only installs the UI's dependencies.

## Folder ownership (keeps merges conflict-free)

- API/UI team: `api/`, `ui/`
- Model work: `src/pathsnap/`, `scripts/`, `results/`, `configs/`, `artifacts/`

## Endpoints

### `GET /health`
```json
{"status": "ok", "model": "stub", "n_trips": 10}
```

### `GET /trips`
Trip ids are found by **scanning the folder in the `PATHSNAP_EVAL_DIR` env var** with
`pathsnap.io.list_trips(EVAL_DIR)`. It handles both one-sub-folder-per-trip (like
`student_test`) and all files flat in one folder, and returns one path per trip; pass that
path to `predict_trip`. **Never hardcode trip ids**: the evaluator's ten trips are ones we have
never seen. The evaluator folder arrives after lunch. Copy it to laptop 1 and point
`PATHSNAP_EVAL_DIR` at it, but don't open or analyse it (evaluation data must stay untouched).
```json
{"trips": [{"trip_id": "00000000", "n_points": 3958}, {"trip_id": "00000007", "n_points": 1206}]}
```

### `GET /models`
Every model we trained, including the weaker ones, with its recorded metrics. Use it for a
leaderboard and a model picker in the UI. `"default"` is the final model.
```json
{"models": [
  {"name": "default", "model": "lightgbm", "description": "final model + HMM", "cv_top1": 0.93, "test_top1": 0.93},
  {"name": "distance_gaussian", "family": "Classic baseline (distance only)", "cv_top1": 0.55, "test_top1": 0.53},
  {"name": "logreg", "family": "Linear", "cv_top1": 0.92, "test_top1": 0.92}
]}
```

### `GET /experiments`
Past runs for a "history" view: the model comparison on the current and the earlier feature
set, the HMM ablation (classifier-only vs + HMM vs classic HMM) and every Bayesian-optimisation
trial. Returns `pathsnap.pipeline.experiment_history()` as-is (lists of table rows).

### `POST /predict`
Request: `{"trip_id": "00000000"}`, optionally with `"model": "<name from /models>"` (default
`"default"`) and `"hmm": false` to show the classifier-only route. Running the same trip with a
weak model, then the final model, makes the improvement visible during the demo.

Response:
```json
{
  "trip_id": "00000000",
  "n_points": 3958,
  "predicted_arcs": [812, 812, 813, null],
  "route_arcs": [812, 813, 977],
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {"type": "Feature", "properties": {"layer": "gps"},
       "geometry": {"type": "MultiPoint", "coordinates": [[12.44, 49.56], [12.45, 49.56]]}},
      {"type": "Feature", "properties": {"layer": "predicted_route"},
       "geometry": {"type": "MultiLineString", "coordinates": [[[12.44, 49.56], [12.45, 49.57]]]}}
    ]
  },
  "metrics": {
    "top1_accuracy": 0.93,
    "coverage_precision": 0.97, "coverage_recall": 0.99, "coverage_f1": 0.98,
    "sequence_precision": 0.96, "sequence_recall": 0.98, "sequence_f1": 0.97,
    "route_mismatch": 0.04,
    "route_continuity": 0.99,
    "n_transitions": 412,
    "n_points_scored": 3901
  },
  "n_route_pieces": 1,
  "timings_ms": {"load": 120, "candidates": 300, "features": 250, "classifier": 90, "hmm": 800,
                 "route": 40, "continuity": 60, "response": 30, "total": 1700}
}
```
- `predicted_arcs`: one entry per GPS point, in order; `null` means unmatched.
- `route_arcs`: the predicted route as arc ids in travel order (all pieces concatenated).
- The route is split into pieces (separate lines in the MultiLineString) at GPS gaps, so no
  line is drawn across hours without evidence. `n_route_pieces` says how many there are.
- **Metrics, in plain words** (show these labels in the UI):
  - `top1_accuracy`: share of GPS points matched to the right road segment.
  - `coverage_*`: did we cover the right roads? Length-weighted, ignores order.
  - `sequence_*`: did we recover the journey? Order- and repeat-aware, so a wrong order or a
    missed lap lowers it.
  - `route_continuity`: share of segment changes a car could physically make. `null` when there
    is nothing to judge (see `n_transitions`).
  - Any metric can be `null` when it cannot be computed (e.g. no `.route` file); show "n/a".
- `timings_ms.total` is the wall clock of the prediction call. Scoring is timed separately;
  log both in the API console.
- GeoJSON coordinates are `[lon, lat]` (the GeoJSON standard). Folium/Leaflet `PolyLine` wants
  `[lat, lon]`, so either use `folium.GeoJson` / `L.geoJSON` directly or swap the order.
- Metric values may be `null` while the model is still a stub.
- Errors: `404 {"detail": "unknown trip_id"}`, `500 {"detail": "<message>"}`.

## Calling the model

The API only needs these functions, all in `pathsnap.pipeline`:

```python
from pathsnap.pipeline import predict_trip, score_trip, list_models, experiment_history

pred = predict_trip(trip_dir, model=None, use_hmm=True)  # model=None -> the final model
metrics = score_trip(trip_dir, pred)                      # server-side only: reads .route
```

`predict_trip` returns every `/predict` response key except `metrics` (plus `model` and `hmm`).
Call `predict_trip` once at startup (warm-up) so the first evaluator request isn't slow.
Until the real pipeline is merged, put this stub in `api/stub_pipeline.py` and import from it:

```python
import time
from pathlib import Path

import pandas as pd


def predict_trip(trip_dir):
    trip_dir = Path(trip_dir)
    t0 = time.perf_counter()
    track = pd.read_csv(trip_dir / f"{trip_dir.name}.track", sep=r"\s+", header=None).to_numpy()
    coords = track[:, :2].tolist()  # [lon, lat]
    return {
        "trip_id": trip_dir.name,
        "n_points": len(track),
        "predicted_arcs": [None] * len(track),
        "route_arcs": [],
        "geojson": {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"layer": "gps"},
             "geometry": {"type": "MultiPoint", "coordinates": coords}},
            {"type": "Feature", "properties": {"layer": "predicted_route"},
             "geometry": {"type": "MultiLineString", "coordinates": [coords]}},  # placeholder: raw GPS
        ]},
        "timings_ms": {"total": round((time.perf_counter() - t0) * 1000, 1)},
    }


def score_trip(trip_dir, prediction):
    return {"top1_accuracy": None, "coverage_f1": None, "sequence_f1": None,
            "route_continuity": None, "n_points_scored": 0}
```

## Rules from the brief (these get checked)

1. **The reference route never leaves laptop 1.** Prediction reads only `.track`, `.nodes`
   and `.arcs`. `score_trip` reads `.route` *after* prediction. The response carries metrics
   only, never the reference route or its geometry.
2. **Log every request to the console** (trip id, number of points, timings, metrics). The
   evaluator will ask to see the API window on laptop 1, and the report needs screenshots of it.
3. The UI must work for **any of the ten** trips listed by `/trips`, with no hardcoded outputs.
   Every result must come through the API.
4. Load the model once at startup (FastAPI lifespan), not per request.

## Networking checklist (test this early, not on demo day)

- Run `uvicorn api.main:app --host 0.0.0.0 --port 8000` on laptop 1.
- Allow inbound TCP 8000 in Windows Defender Firewall on laptop 1 (needs admin).
- Campus Wi-Fi often blocks device-to-device traffic. A phone hotspot with both laptops on it
  is the reliable fallback.
- From laptop 2, run `curl http://<laptop-1-IP>:8000/health`. Get laptop 1's IP from `ipconfig`.
- For a browser-based UI, add FastAPI `CORSMiddleware`.

## Rehearsal data

Point `PATHSNAP_EVAL_DIR` at `Provided Material/Student folder/student_test`. It has the same
layout as the evaluator's set, including `.route` files for scoring.
