# PathSnap backend — run guide

FastAPI layer between the ML pipeline (`pathsnap`) and the evaluator UI. It exposes five
endpoints (`/health`, `/trips`, `/predict`, `/models`, `/experiments`), loads the model once at
startup, scores every prediction server-side and logs every request to the console.

## Install (laptop 1)

From this folder (`branches/secure-backend`):

```powershell
uv pip install --python ..\..\.venv -r requirements.txt
uv pip install --python ..\..\.venv -e ..\..\pathsnap
```

`pathsnap` brings the ML dependencies; `requirements.txt` is only the web layer. The model
bundle (`pathsnap\artifacts\pathsnap_model.joblib`) must exist — it ships with the repo.

## Run on laptop 1

```powershell
..\..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Allow inbound TCP 8000 once (admin PowerShell):

```powershell
New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

The startup banner prints the model, the trip folder, the number of trips and the exact URL to
type into the UI, e.g. `http://192.168.1.23:8000`.

## Connect laptop 2

1. Start the UI (`branches/frontend/ui`, `npm run dev`).
2. Type the banner's `http://<laptop-1-ip>:8000` into the UI's **API address** box.
3. If it cannot connect, put both laptops on a phone hotspot — college Wi-Fi often blocks
   device-to-device traffic. Windows Firewall must allow port 8000 on laptop 1.

Optional LAN discovery (off by default), with the same `DISCOVERY_SECRET` (64 hex chars =
32 bytes) in both laptops' `.env`:

```powershell
# laptop 1
$env:PATHSNAP_DISCOVERY="true"; $env:DISCOVERY_SECRET="<64 hex chars>"
# laptop 2, same secret
..\..\.venv\Scripts\python.exe -m networking.discovery
# prints: PathSnap server found: http://<ip>:8000
```

## Settings (`.env`, see `.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `0.0.0.0` | uvicorn bind address (informational; the CLI flag wins) |
| `PORT` | `8000` | shown in the banner and used by discovery |
| `ALLOWED_ORIGINS` | `*` | comma-separated CORS origins; `*` for the unknown UI origin |
| `RATE_LIMIT` | `60/minute` | slowapi default for decorated routes |
| `PATHSNAP_EVAL_DIR` | auto-detect `Provided Material/Student folder/student_test` | folder with one sub-folder per trip (or flat trip files) |
| `PATHSNAP_REQUIRE_AUTH` | `false` | when true, the five ML endpoints need a bearer token from `/auth/pair` |
| `PATHSNAP_DISCOVERY` | `false` | start the UDP broadcaster |
| `DISCOVERY_SECRET` | empty | required for discovery; 32 bytes as hex |
| `APP_ENV` | `development` | free-form label in logs |

Auth is **off** for the demo (the UI sends no token) and the API is plain HTTP. Keep
`security/generate_certs.py` out of startup; it is not wired in.

## Demo day

Copy the evaluator's folder to laptop 1, then either set

```powershell
$env:PATHSNAP_EVAL_DIR="C:\path\to\evaluator_trips"
```

and restart the API, or leave the default for rehearsal. Do not open or analyse the evaluator
folder: reference routes must stay on laptop 1, and the API never returns them.

## Endpoints

- `GET /health` → `{"status":"ok","model":"lightgbm_tuned","n_trips":10}`
- `GET /trips` → `{"trips":[{"trip_id":"00000000","n_points":2503}, …]}`
- `POST /predict` `{"trip_id":"00000052","model":null,"hmm":true}` → prediction + `metrics`
- `GET /models` → leaderboard of every saved model
- `GET /experiments` → comparison tables, HMM ablation, tuning trials

Errors: `404 {"detail":"unknown trip_id"}`, `404` for an unknown model, `422` for a malformed
trip id, `500 {"detail":"prediction failed on the server"}` (the traceback stays in the console).

## Tests

```powershell
..\..\.venv\Scripts\python.exe -m pytest -q
```

The suite uses the real `student_test` folder and the real model: endpoint contract, the
reference-route leak check, unknown trip/model/path handling, classifier-only mode, auth on/off
and the discovery HMAC/nonce checks. A prediction takes ~0.2–1.2 s depending on trip size; calls
are serialised with a lock because the pipeline is CPU-bound.
