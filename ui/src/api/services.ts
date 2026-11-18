/** The three endpoints defined in API_CONTRACT.md. */
import { PREDICT_TIMEOUT_MS } from '@/config'
import { request } from './client'
import type { HealthResponse, PredictResponse, TripsResponse } from './types'
import { validateHealth, validatePredict, validateTrips, type Validated } from './validate'

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request({ url: '/health', method: 'GET', signal }, validateHealth)
}

export function getTrips(signal?: AbortSignal): Promise<TripsResponse> {
  return request({ url: '/trips', method: 'GET', signal }, validateTrips)
}

export function postPredict(tripId: string, signal?: AbortSignal): Promise<Validated<PredictResponse>> {
  return request(
    {
      url: '/predict',
      method: 'POST',
      data: { trip_id: tripId },
      headers: { 'Content-Type': 'application/json' },
      timeout: PREDICT_TIMEOUT_MS,
      signal,
      logNote: `trip ${tripId}`,
    },
    (data) => validatePredict(data, tripId),
  )
}
