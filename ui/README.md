# PathSnap UI

The evaluator interface for PathSnap. It runs on **Laptop 2**, talks to the FastAPI backend on **Laptop 1** over HTTP, and shows each trip's GPS track, the reconstructed route and the evaluation metrics the API returns.

The UI does not run any ML. It shows only what the API sends back. It computes no evaluation metrics, has no hardcoded trips or results, and never receives reference routes.

## Stack

React 19, Vite, TypeScript, Tailwind CSS v4, shadcn/ui (Radix), Lucide icons, React Leaflet, Recharts and Axios. Fonts (Barlow, Barlow Condensed) are bundled locally, so the UI itself needs no internet. Only the basemap tiles do.

## Setup

Requires Node.js 20 or later (tested with Node 22).

```bash
cd ui
npm install
npm run dev                     # http://localhost:5173, also reachable on the LAN
```

Create `ui/.env.local` (git-ignored) with Laptop 1's address:

```
VITE_API_BASE_URL=http://192.168.1.20:8000
```

| Command | What it does |
|---|---|
| `npm run dev` | Development server on port 5173, bound to all interfaces |
| `npm run build` | Type-check and build to `dist/` |
| `npm run preview` | Serve the built `dist/` on port 4173 (use this on demo day) |
| `npm run lint` | oxlint |

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend address. `localhost` is only right when API and UI run on the same machine. |
| `VITE_PREDICT_TIMEOUT_MS` | `180000` | Timeout for `POST /predict` |
| `VITE_TILE_URL` / `VITE_TILE_ATTRIBUTION` | OpenStreetMap standard tiles | Basemap source |
| `VITE_TILE_DARKEN` | `true` | Darken the tiles with a CSS filter. Set to `false` for a tile style that is already dark. |

**Changing the API address without rebuilding.** Click the API status button in the top bar, enter `http://<laptop-1-ip>:8000` and choose **Connect**. The address is saved in this browser and takes priority over `.env.local`. **Reset** (or **Use .env address**) clears it.

## API contract (v0.1)

The UI uses exactly the three endpoints in `API_CONTRACT.md`. The types are in `src/api/types.ts`.

| Endpoint | Used for |
|---|---|
| `GET /health` | Connection status, model name (shows a warning when it is `stub`), trip count. Polled every 15 s. |
| `GET /trips` | The trip list. Never hardcoded. |
| `POST /predict {"trip_id"}` | Map layers, the match ribbon, route arcs, metrics and timings |

Every response is checked at runtime (`src/api/validate.ts`):

- **Wrong shape:** an unexpected response type, or a `trip_id` that differs from the one requested, raises an error. The UI does not render the data.
- **Minor gaps:** a missing metric or timing key, or `predicted_arcs` with a different length from `n_points`, are shown as response notes beside the result.
- **Null metrics** are shown as **N/A** and are never replaced.

**Map layers** come from `geojson.features[].properties.layer`:

- `gps` (points) and `predicted_route` (line) are expected.
- `road_network` is drawn when the API includes it. Otherwise the legend says it is not included.
- Any other layer name is drawn as an extra layer.
- A layer whose name suggests reference or ground-truth data is refused and never drawn.

**Stub output.** When every `predicted_arcs` entry is `null` and `route_arcs` is empty, the UI labels the result as stub placeholder output. It draws the raw GPS line dashed and grey, not as a reconstructed route.

**Errors.** Each failure gets its own message with a retry option: network or CORS failure, timeout, `404 {"detail"}`, `500 {"detail"}`, a rejected request, and an invalid response.

## Two-laptop setup

**Laptop 1 (API, set up by the backend team):**

1. Run `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
2. Allow inbound TCP 8000 in Windows Defender Firewall.
3. Enable CORS. The browser on Laptop 2 is a different origin, so without CORS every request fails as "API unreachable". Example:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
   ```

**Laptop 2 (this UI):**

1. Check the network: `curl http://<laptop-1-ip>:8000/health`. If campus Wi-Fi blocks device-to-device traffic, put both laptops on a phone hotspot.
2. Set `VITE_API_BASE_URL` in `.env.local`, or set the address in the UI.
3. Run `npm run build` and then `npm run preview`. Open `http://localhost:4173`.

## Pages

| Page | Contents |
|---|---|
| **Overview** | Status strip (API, model, trips from `/health` and `/trips`), a schematic of map-matching (labelled as an illustration, not trip data), the two-laptop pipeline with real stage timings after a prediction, dataset partitions (static, from the brief) and the integrity rules |
| **Trip explorer** | The main evaluator screen. Trip list with search, status filter and sort. Map with zoom, fit-trip, full screen, scale bar, cursor coordinates, a Dark/Light/Off basemap switch and layer toggles that double as the legend. Matched GPS points are filled dots, unmatched points are hollow rings, and Start and End are marked. The inspector shows trip facts, live elapsed time while predicting, server stage timings and metric tiles |
| **Route analysis** | Trip switcher, sequence counts, the match ribbon with a 48-point detail strip and previous/next-gap navigation, a point inspector (arc, run, position in `route_arcs`), arc runs you can filter and sort, and route arcs that jump to their first matched GPS point |
| **Model analytics** | Metric cards with explanations, a stage timing table with shares of the server total, prediction-output counts kept separate from metrics, the per-trip session comparison as a chart or a table, and what the API does not serve |
| **Evaluation** | Evaluator walkthrough, batch runs (all trips, trips not run yet, or failed trips) with per-trip progress, a sortable and filterable results table with inline Retry and error details, and a console-style request log you can filter and copy |

**The match ribbon** draws one cell per GPS observation, in time order. Each run of consecutive points on the same arc is one block, and the shade alternates at every arc change. Hatching means unmatched (`null`). Hover to read a point; click it, or use the arrow keys, to select it. The selected point is ringed on the map, together with the other points of its run.

**Missing versus zero.** A null metric is shown as **N/A** on a dashed tile. A real `0` from the API is shown as `0`. Stage timings the API did not report are listed as N/A.

**Placeholder output.** When every `predicted_arcs` entry is null and `route_arcs` is empty (the stub), the result is badged **Placeholder**. GPS points are drawn as plain dots with "none matched", and the raw GPS line is drawn dashed grey, never as a route.

**Derived counts.** Matched and unmatched counts, gaps, arc runs and distinct arcs describe the prediction itself and are labelled that way. Accuracy, precision, recall, F1 and continuity come only from the API. The **mean** row on the Evaluation page is an unweighted average of the per-trip values the API returned.

**Feedback.** Toasts confirm each prediction (with server and round-trip time), failure, cancellation and batch outcome. Nothing is stored between page reloads except the selected trip, the API address and the basemap choice.

## Backend enhancements that would unlock more UI

None of these are implemented. They would need new fields or endpoints in the API, so they are listed here instead of faked:

- Per-point timestamps in `/predict`, for speed and time-gap views on the ribbon
- Per-point candidate scores or classifier probabilities, for a candidate inspector
- Classifier-only predictions alongside the decoded route, for the Section 5 comparison
- A read-only endpoint serving the model team's recorded results, for the five-model comparison and the optimisation trials
- Trip metadata in `/trips` (duration, bounding box, sampling interval), to describe a trip before it is predicted

## What the UI does not show, and why

- **The five-model comparison, cross-validation scores and Bayesian optimisation trials.** API v0.1 does not serve them. They belong in `results/` and the technical report. The Model analytics page says so instead of showing invented numbers.
- **Classifier-only versus HMM/Viterbi comparison.** The API returns only the final decoded output. Report Section 5 covers this comparison.
- **The Folium maps the brief requires.** This UI uses React Leaflet for the live evaluator view, which does **not** replace the mandatory Folium visualisations. The Folium map-generation code and its HTML output are part of the shared model/report work (`scripts/` and `results/`). This folder does not contain or change them.

## Testing done

These checks ran against a local test double of the API. The test double implements `API_CONTRACT.md`, including the contract's stub pipeline. It is not part of this repo.

- `npm run build` and `tsc -b` pass. Lint reports only fast-refresh and effect-style warnings.
- Browser tests (Playwright driving Chrome) passed:
  - every page with the API offline, with the stub model, and with a model that returns arcs
  - running all 10 student-test trips in a batch, including one forced 500 error with inline Retry
  - ribbon click, next-gap navigation, the 48-cell detail strip and run filtering
  - trip search
  - a 500 error on one trip, an unreachable API and an unknown trip (404)
- Layouts were checked at 1920×1080, 1440×900, 1366×768 and 1280×720 (projector). At projector sizes the map, ribbon and inspector fit without page scroll. At 390 px wide no page scrolls sideways.
- No console errors appeared except the browser's own logs of the deliberately failed requests.

**Still to test against the team's real backend:**

- [ ] Real `/predict` output from the trained pipeline: route geometry, real metrics, all stage timings
- [ ] CORS enabled on the real API
- [ ] Remote connection from Laptop 2 to Laptop 1 over the demo network
- [ ] Prediction times on the largest trips, compared with the 180 s timeout
- [ ] Whether the API sends a `road_network` layer, and how many vertices it has

## Folder structure

```
src/
  api/          types, runtime validation, axios client + request log, endpoint services
  state/        session context: health, trips, selection, predictions, batch run, focused point
  components/
    layout/     app shell, sidebar rail, top bar, API status and address setting
    map/        RouteMap and imperative Leaflet layers (canvas GPS points, panes, controls, fitting)
    ribbon/     MatchRibbon (with detail strip) and PointDetails
    metrics/    metric tiles, stage strip, timing table, per-trip session chart
    trips/      trip list, prediction panel, trip context bar
    evaluation/ request console
    overview/   map-matching schematic
    common/     panels, notices, empty states, status dots, info tips, segmented control, toasts
    ui/         shadcn/ui components
  lib/          GeoJSON layers, prediction view, sequence statistics, trip status, toasts, formatting
  pages/        Overview, TripExplorer, RouteAnalysis, ModelAnalytics, Evaluation
```

## Design tokens

Colours are defined in `src/index.css` and are dark only: a layered blue-grey slate, darkest at the navigation rail.

| Role | Colour |
|---|---|
| Navigation rail / page / inset | `#161d27` / `#1c2431` / `#171f29` |
| Panels / raised / borders | `#242e3c` / `#2d3949` / `#3a4759` |
| Text | `#eef3f8` (12.3:1 on panels), `#b0bdcc` (7.2:1), `#8796a8` (4.5:1) |
| Predicted route (the one accent) | cyan `#38d3df` |
| GPS points | matched blue `#5c9dff`, unmatched hollow `#e9eef5` |
| Road network | `#7b889a` |
| Status | `#5fd394` (ok) / `#f0b955` (warning) / `#f2707a` (error) |
| Charts | `#4a90ec`, `#1ba878` |

The status colours always appear with an icon and a label. The chart pair was checked for colour-blind separation and contrast on the panel colour. The sidebar collapses to icons below 1600 px so the map keeps its width at projector resolutions.
