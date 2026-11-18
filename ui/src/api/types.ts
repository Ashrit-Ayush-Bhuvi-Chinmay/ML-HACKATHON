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
 * which never leaves Laptop 1. Any value may be null (e.g. while the model is a stub).
 */
export interface PredictionMetrics {
  top1_accuracy: number | null
  route_precision: number | null
  route_recall: number | null
  route_f1: number | null
  route_continuity: number | null
  n_points_scored: number | null
}

/**
 * Stage timings in milliseconds. The contract lists load, candidates, features,
 * classifier, hmm and total; the stub only returns total, so every key is optional.
 */
export type TimingsMs = Partial<Record<'load' | 'candidates' | 'features' | 'classifier' | 'hmm' | 'total', number>> &
  Record<string, number>

/** POST /predict response */
export interface PredictResponse {
  trip_id: string
  n_points: number
  /** One entry per GPS point, in order. null = unmatched. */
  predicted_arcs: (number | null)[]
  /** Connected predicted route as arc ids in travel order. */
  route_arcs: number[]
  geojson: FeatureCollection
  metrics: PredictionMetrics
  timings_ms: TimingsMs
}

/** FastAPI error body: {"detail": "..."} */
export interface ApiErrorBody {
  detail: string | unknown
}
