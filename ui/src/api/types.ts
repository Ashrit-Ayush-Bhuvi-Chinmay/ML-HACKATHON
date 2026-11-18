/**
 * Types for the PathSnap API, mirroring API_CONTRACT.md (v0.1).
 * Keep these in sync with the FastAPI response models in api/.
 */

/** GET /health */
export interface HealthResponse {
  status: string
  /** Model identifier reported by the API. "stub" means the placeholder pipeline is loaded. */
  model: string
  n_trips: number
}

/** One entry of GET /trips */
export interface TripSummary {
  trip_id: string
  n_points: number
}

/** GET /trips */
export interface TripsResponse {
  trips: TripSummary[]
}

/** POST /predict request body */
export interface PredictRequest {
  trip_id: string
  model?: string | null
  hmm?: boolean
}

/** GeoJSON (RFC 7946). Coordinates are [longitude, latitude]. */
export type Position = [number, number] | [number, number, number]

export type Geometry =
  | { type: 'Point'; coordinates: Position }
  | { type: 'MultiPoint'; coordinates: Position[] }
  | { type: 'LineString'; coordinates: Position[] }
  | { type: 'MultiLineString'; coordinates: Position[][] }
  | { type: 'Polygon'; coordinates: Position[][] }
  | { type: 'MultiPolygon'; coordinates: Position[][][] }

export interface Feature {
  type: 'Feature'
  /** properties.layer names the map layer: "gps", "predicted_route", optionally "road_network". */
  properties: ({ layer?: string } & Record<string, unknown>) | null
  geometry: Geometry | null
}

export interface FeatureCollection {
  type: 'FeatureCollection'
  features: Feature[]
}

/**
 * Evaluation metrics computed server-side by score_trip() against the reference route,
 * which never leaves Laptop 1. Any value may be null.
 */
export interface PredictionMetrics {
  /** Share of GPS points matched to the right road segment. */
  top1_accuracy: number | null
  /** Did we cover the right roads? Length-weighted, ignores order. */
  coverage_precision: number | null
  coverage_recall: number | null
  coverage_f1: number | null
  /** Did we recover the journey? Order- and lap-aware, the main route score. */
  sequence_precision: number | null
  sequence_recall: number | null
  sequence_f1: number | null
  /** Wrongly added + missed route length as a fraction of the true route. Lower is better. */
  route_mismatch: number | null
  /** Share of segment changes a car could physically make; null when nothing to judge. */
  route_continuity: number | null
  /** How many segment changes the continuity is based on. */
  n_transitions: number | null
  /** Points with a reliable reference label. */
  n_points_scored: number | null
  /** Median GPS-to-reference distance in metres, when the API reports it. */
  reference_median_dist_m?: number | null
  /** Optional explanation, e.g. "no reference route for this trip; prediction only". */
  note?: string
}

/**
 * Stage timings in milliseconds. The contract lists load, candidates, features,
 * classifier, hmm, route, continuity and response; classifier-only runs report `decode`
 * instead of `hmm`. Unknown extra keys are tolerated.
 */
export type TimingsMs = Partial<
  Record<
    'load' | 'candidates' | 'features' | 'classifier' | 'hmm' | 'decode' | 'route' | 'continuity' | 'response' | 'total',
    number
  >
> &
  Record<string, number>

/** POST /predict response */
export interface PredictResponse {
  trip_id: string
  n_points: number
  /** One entry per GPS point, in order. null = unmatched. */
  predicted_arcs: (number | null)[]
  /** The predicted route as arc ids in travel order, split into pieces at GPS gaps. */
  route_arcs: number[]
  geojson: FeatureCollection
  metrics: PredictionMetrics
  timings_ms: TimingsMs
  /** Number of lines in the predicted route (the route is split at GPS gaps). */
  n_route_pieces?: number
  /** GPS points with no matched arc. */
  n_unmatched?: number
  /** Model name reported by the API for this prediction. */
  model?: string
  /** Whether HMM/Viterbi decoding produced this route. */
  hmm?: boolean
  /** Systematic GPS-shift correction applied by the API, if any. */
  offset_correction?: { applied?: boolean; dx_m?: number; dy_m?: number } | null
}

/** FastAPI error body: {"detail": "..."} */
export interface ApiErrorBody {
  detail: string | unknown
}

/** GET /models — every saved model and its recorded metrics. Only `name` is guaranteed. */
export interface ModelInfo {
  name: string
  model?: string
  family?: string
  description?: string
  cv_top1?: number | null
  cv_macro_f1_mean?: number | null
  test_top1?: number | null
  test_macro_f1?: number | null
  test_sequence_f1?: number | null
  test_coverage_f1?: number | null
  test_plausible_continuity?: number | null
  test_log_loss?: number | null
  infer_ms_per_1k_points?: number | null
  [key: string]: unknown
}

export interface ModelsResponse {
  models: ModelInfo[]
}

export type ExperimentRow = Record<string, unknown>

/** GET /experiments — comparison tables, HMM ablations and tuning trials. */
export interface ExperimentsResponse {
  model_comparison: ExperimentRow[]
  model_comparison_featureset_v1: ExperimentRow[]
  hmm_ablation: Record<string, ExperimentRow[]>
  tuning_trials: Record<string, ExperimentRow[]>
}

/** Optional POST /predict body fields. */
export interface PredictOptions {
  model?: string | null
  hmm?: boolean
}
