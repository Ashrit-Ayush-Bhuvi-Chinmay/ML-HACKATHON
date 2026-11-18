/**
 * Runtime validation of API responses.
 *
 * Fatal problems (wrong shape) throw InvalidResponseError so the UI never renders
 * half-understood data. Non-fatal problems (e.g. a missing optional timing key)
 * are returned as warnings and shown next to the result.
 */
import type {
  Feature,
  FeatureCollection,
  HealthResponse,
  PredictionMetrics,
  PredictResponse,
  TimingsMs,
  TripsResponse,
} from './types'

export class InvalidResponseError extends Error {
  constructor(endpoint: string, problem: string) {
    super(`${endpoint} returned an unexpected response: ${problem}`)
    this.name = 'InvalidResponseError'
  }
}

export interface Validated<T> {
  value: T
  warnings: string[]
}

type Obj = Record<string, unknown>

const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const isInt = (v: unknown): v is number => isNum(v) && Number.isInteger(v)

export function validateHealth(data: unknown): HealthResponse {
  const ep = 'GET /health'
  if (!isObj(data)) throw new InvalidResponseError(ep, 'body is not a JSON object')
  if (typeof data.status !== 'string') throw new InvalidResponseError(ep, '"status" is missing')
  return {
    status: data.status,
    model: typeof data.model === 'string' ? data.model : 'unknown',
    n_trips: isInt(data.n_trips) ? data.n_trips : 0,
  }
}

export function validateTrips(data: unknown): TripsResponse {
  const ep = 'GET /trips'
  if (!isObj(data) || !Array.isArray(data.trips)) throw new InvalidResponseError(ep, '"trips" is not a list')
  const trips = data.trips.map((t, i) => {
    if (!isObj(t) || typeof t.trip_id !== 'string' || !t.trip_id) {
      throw new InvalidResponseError(ep, `trip #${i + 1} has no "trip_id"`)
    }
    return { trip_id: t.trip_id, n_points: isInt(t.n_points) ? t.n_points : 0 }
  })
  return { trips }
}

const METRIC_KEYS: (keyof PredictionMetrics)[] = [
  'top1_accuracy',
  'route_precision',
  'route_recall',
  'route_f1',
  'route_continuity',
  'n_points_scored',
]

function validateMetrics(raw: unknown, warnings: string[]): PredictionMetrics {
  const out = {} as PredictionMetrics
  if (!isObj(raw)) {
    warnings.push('Response has no "metrics" object; all metrics shown as N/A.')
  }
  const src: Obj = isObj(raw) ? raw : {}
  for (const k of METRIC_KEYS) {
    const v = src[k]
    if (v === null || v === undefined) {
      out[k] = null
    } else if (isNum(v)) {
      out[k] = v
    } else {
      warnings.push(`Metric "${k}" is not a number; shown as N/A.`)
      out[k] = null
    }
  }
  return out
}

function validateTimings(raw: unknown, warnings: string[]): TimingsMs {
  if (!isObj(raw)) {
    warnings.push('Response has no "timings_ms" object.')
    return {} as TimingsMs
  }
  const out: Record<string, number> = {}
  for (const [k, v] of Object.entries(raw)) {
    if (isNum(v)) out[k] = v
    else warnings.push(`Timing "${k}" is not a number and was ignored.`)
  }
  return out as TimingsMs
}

function validateGeojson(raw: unknown, ep: string): FeatureCollection {
  if (!isObj(raw) || raw.type !== 'FeatureCollection' || !Array.isArray(raw.features)) {
    throw new InvalidResponseError(ep, '"geojson" is not a GeoJSON FeatureCollection')
  }
  const features = raw.features.filter(
    (f): f is Feature => isObj(f) && f.type === 'Feature' && (f.geometry === null || isObj(f.geometry)),
  )
  return { type: 'FeatureCollection', features }
}

export function validatePredict(data: unknown, requestedTripId: string): Validated<PredictResponse> {
  const ep = 'POST /predict'
  const warnings: string[] = []
  if (!isObj(data)) throw new InvalidResponseError(ep, 'body is not a JSON object')
  if (typeof data.trip_id !== 'string') throw new InvalidResponseError(ep, '"trip_id" is missing')
  if (data.trip_id !== requestedTripId) {
    throw new InvalidResponseError(ep, `asked for trip ${requestedTripId} but received trip ${data.trip_id}`)
  }
  if (!isInt(data.n_points)) throw new InvalidResponseError(ep, '"n_points" is not an integer')
  if (!Array.isArray(data.predicted_arcs)) throw new InvalidResponseError(ep, '"predicted_arcs" is not a list')
  if (!Array.isArray(data.route_arcs)) throw new InvalidResponseError(ep, '"route_arcs" is not a list')

  const predicted_arcs = data.predicted_arcs.map((a) => (isInt(a) ? a : null))
  const badPredicted = data.predicted_arcs.filter((a) => a !== null && !isInt(a)).length
  if (badPredicted > 0) warnings.push(`${badPredicted} predicted arc value(s) were not integers and are treated as unmatched.`)
  if (predicted_arcs.length !== data.n_points) {
    warnings.push(
      `"predicted_arcs" has ${predicted_arcs.length} entries but n_points is ${data.n_points}; the contract expects one per GPS point.`,
    )
  }

  const route_arcs = data.route_arcs.filter(isInt)
  if (route_arcs.length !== data.route_arcs.length) warnings.push('Some "route_arcs" values were not integers and were dropped.')

  return {
    value: {
      trip_id: data.trip_id,
      n_points: data.n_points,
      predicted_arcs,
      route_arcs,
      geojson: validateGeojson(data.geojson, ep),
      metrics: validateMetrics(data.metrics, warnings),
      timings_ms: validateTimings(data.timings_ms, warnings),
    },
    warnings,
  }
}
