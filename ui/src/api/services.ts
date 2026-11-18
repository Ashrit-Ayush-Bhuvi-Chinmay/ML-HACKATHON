/** The endpoints defined in API_CONTRACT.md. */
import { PREDICT_TIMEOUT_MS } from '@/config'
import { request } from './client'
import type { ExperimentsResponse, HealthResponse, ModelsResponse, PredictOptions, PredictRequest, PredictResponse, TripsResponse } from './types'
import { validateExperiments, validateHealth, validateModels, validatePredict, validateTrips, type Validated } from './validate'

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request({ url: '/health', method: 'GET', signal }, validateHealth)
}

export function getTrips(signal?: AbortSignal): Promise<TripsResponse> {
  return request({ url: '/trips', method: 'GET', signal }, validateTrips)
}

export function getModels(signal?: AbortSignal): Promise<ModelsResponse> {
  return request({ url: '/models', method: 'GET', signal }, validateModels)
}

export function getExperiments(signal?: AbortSignal): Promise<ExperimentsResponse> {
  return request({ url: '/experiments', method: 'GET', signal }, validateExperiments)
}

export function postPredict(
  tripId: string,
  options: PredictOptions = {},
  signal?: AbortSignal,
): Promise<Validated<PredictResponse>> {
  const data: PredictRequest = { trip_id: tripId }
  if (options.model) data.model = options.model
  if (options.hmm !== undefined) data.hmm = options.hmm
  const notes = [`trip ${tripId}`]
  if (options.model) notes.push(options.model)
  if (options.hmm === false) notes.push('no HMM')

  return request(
    {
      url: '/predict',
      method: 'POST',
      data,
      headers: { 'Content-Type': 'application/json' },
      timeout: PREDICT_TIMEOUT_MS,
      signal,
      logNote: notes.join(' · '),
    },
    (body) => validatePredict(body, tripId),
  )
}
