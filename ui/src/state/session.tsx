/**
 * Session state shared by every page: backend health, trip list, the selected trip,
 * prediction results per trip and the GPS point currently in focus (map <-> ribbon).
 *
 * Everything shown in the UI comes from these API responses. Nothing is cached across
 * page reloads except the selected trip id and the API address.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ApiError, setApiBaseUrl } from '@/api/client'
import { getHealth, getTrips, postPredict } from '@/api/services'
import type { HealthResponse, PredictResponse, TripSummary } from '@/api/types'
import { HEALTH_POLL_MS, resolveApiBaseUrl, saveApiBaseUrl, type ApiUrlSource } from '@/config'
import { fmtMs } from '@/lib/format'
import { toast } from '@/lib/toast'

export type HealthState =
  | { status: 'checking'; data?: HealthResponse; checkedAt?: number }
  | { status: 'online'; data: HealthResponse; checkedAt: number; latencyMs: number }
  | { status: 'offline'; error: ApiError; checkedAt: number; data?: undefined }

export type TripsState =
  | { status: 'idle' | 'loading'; trips: TripSummary[]; error?: undefined }
  | { status: 'ready'; trips: TripSummary[]; loadedAt: number; error?: undefined }
  | { status: 'error'; trips: TripSummary[]; error: ApiError }

export interface PredictionResult {
  response: PredictResponse
  warnings: string[]
  receivedAt: number
  /** Measured in the browser: request sent -> response parsed. Includes network time. */
  roundTripMs: number
  /** Model name reported by /health when the prediction ran. */
  model: string | null
}

export interface PredictionRecord {
  status: 'running' | 'success' | 'error'
  startedAt: number
  /** Latest successful result (kept while a re-run is in progress or after a failed re-run). */
  result?: PredictionResult
  error?: ApiError
}

export type RunOutcome = 'success' | 'error' | 'cancelled'

export interface BatchState {
  running: boolean
  /** What the batch is doing, for display: "all trips", "trips not run yet", "failed trips". */
  label: string
  queue: string[]
  done: number
  current: string | null
  succeeded: string[]
  failed: string[]
  startedAt: number | null
  finishedAt: number | null
  stopped: boolean
}

const IDLE_BATCH: BatchState = {
  running: false,
  label: '',
  queue: [],
  done: 0,
  current: null,
  succeeded: [],
  failed: [],
  startedAt: null,
  finishedAt: null,
  stopped: false,
}

interface SessionValue {
  apiUrl: string
  apiUrlSource: ApiUrlSource
  setApiUrl: (url: string | null) => void

  health: HealthState
  /** A health request is in flight (status keeps its last value meanwhile). */
  healthChecking: boolean
  refreshHealth: () => void
  isStubModel: boolean

  trips: TripsState
  loadTrips: () => void

  selectedTripId: string | null
  selectTrip: (id: string | null) => void
  selectedTrip: TripSummary | null

  predictions: Record<string, PredictionRecord>
  runPrediction: (tripId: string) => Promise<RunOutcome>
  cancelPrediction: (tripId: string) => void

  batch: BatchState
  /** Runs the given trips one after another (sequential keeps the API log readable). */
  runBatch: (tripIds: string[], label: string) => Promise<void>
  runAllTrips: () => Promise<void>
  cancelBatch: () => void

  focusIndex: number | null
  setFocusIndex: (i: number | null) => void
}

const SessionContext = createContext<SessionValue | null>(null)

const SELECTED_KEY = 'pathsnap.selectedTrip'

function readSelected(): string | null {
  try {
    return window.sessionStorage.getItem(SELECTED_KEY)
  } catch {
    return null
  }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(() => resolveApiBaseUrl(), [])
  const [apiUrl, setApiUrlState] = useState(initial.url)
  const [apiUrlSource, setApiUrlSource] = useState<ApiUrlSource>(initial.source)

  const [health, setHealth] = useState<HealthState>({ status: 'checking' })
  const [healthChecking, setHealthChecking] = useState(false)
  const [trips, setTrips] = useState<TripsState>({ status: 'idle', trips: [] })
  const [storedTripId, setSelectedTripId] = useState<string | null>(readSelected)
  const [predictions, setPredictions] = useState<Record<string, PredictionRecord>>({})
  const [batch, setBatch] = useState<BatchState>(IDLE_BATCH)
  const [focusIndex, setFocusIndex] = useState<number | null>(null)

  const controllers = useRef(new Map<string, AbortController>())
  const batchCancelled = useRef(false)
  const inBatch = useRef(false)
  const healthRef = useRef(health)
  useEffect(() => {
    healthRef.current = health
  }, [health])

  // ---- health -------------------------------------------------------------
  const refreshHealth = useCallback(async () => {
    setHealthChecking(true)
    const t0 = performance.now()
    try {
      const data = await getHealth()
      setHealth({ status: 'online', data, checkedAt: Date.now(), latencyMs: performance.now() - t0 })
    } catch (e) {
      setHealth({ status: 'offline', error: e as ApiError, checkedAt: Date.now() })
    } finally {
      setHealthChecking(false)
    }
  }, [])

  // ---- trips --------------------------------------------------------------
  const loadTrips = useCallback(async () => {
    setTrips((t) => ({ status: 'loading', trips: t.trips }))
    try {
      const res = await getTrips()
      setTrips({ status: 'ready', trips: res.trips, loadedAt: Date.now() })
    } catch (e) {
      setTrips((t) => ({ status: 'error', trips: t.trips, error: e as ApiError }))
    }
  }, [])

  // Initial check + polling. Re-runs when the API address changes.
  useEffect(() => {
    void refreshHealth()
    const id = window.setInterval(() => void refreshHealth(), HEALTH_POLL_MS)
    return () => window.clearInterval(id)
  }, [refreshHealth, apiUrl])

  // Load trips as soon as the backend is reachable, and reload if it came back after an outage.
  const online = health.status === 'online'
  useEffect(() => {
    if (online && (trips.status === 'idle' || trips.status === 'error')) void loadTrips()
  }, [online, trips.status, loadTrips])

  const selectTrip = useCallback((id: string | null) => {
    setSelectedTripId(id)
    setFocusIndex(null)
    try {
      if (id) window.sessionStorage.setItem(SELECTED_KEY, id)
      else window.sessionStorage.removeItem(SELECTED_KEY)
    } catch {
      /* ignore */
    }
  }, [])

  // ---- prediction ---------------------------------------------------------
  const runPrediction = useCallback(async (tripId: string): Promise<RunOutcome> => {
    controllers.current.get(tripId)?.abort()
    const controller = new AbortController()
    controllers.current.set(tripId, controller)
    const startedAt = Date.now()
    setPredictions((p) => ({ ...p, [tripId]: { ...p[tripId], status: 'running', startedAt, error: undefined } }))
    const t0 = performance.now()
    try {
      const { value, warnings } = await postPredict(tripId, controller.signal)
      const h = healthRef.current
      const result: PredictionResult = {
        response: value,
        warnings,
        receivedAt: Date.now(),
        roundTripMs: performance.now() - t0,
        model: h.status === 'online' ? h.data.model : null,
      }
      setPredictions((p) => ({ ...p, [tripId]: { status: 'success', startedAt, result } }))
      if (!inBatch.current) {
        toast('success', `Prediction received for trip ${tripId}`, `Server ${fmtMs(value.timings_ms.total)}, round trip ${fmtMs(result.roundTripMs)}.`)
      }
      return 'success'
    } catch (e) {
      const err = e as ApiError
      setPredictions((p) => ({ ...p, [tripId]: { ...p[tripId], status: 'error', startedAt, error: err } }))
      if (!inBatch.current) {
        if (err.kind === 'cancelled') toast('info', `Prediction cancelled for trip ${tripId}`)
        else toast('error', `Prediction failed for trip ${tripId}`, err.message)
      }
      return err.kind === 'cancelled' ? 'cancelled' : 'error'
    } finally {
      if (controllers.current.get(tripId) === controller) controllers.current.delete(tripId)
    }
  }, [])

  const cancelPrediction = useCallback((tripId: string) => {
    controllers.current.get(tripId)?.abort()
  }, [])

  const runBatch = useCallback(
    async (tripIds: string[], label: string) => {
      if (tripIds.length === 0) return
      batchCancelled.current = false
      inBatch.current = true
      setBatch({ ...IDLE_BATCH, running: true, label, queue: tripIds, startedAt: Date.now() })
      let ok = 0
      let failed = 0
      for (let i = 0; i < tripIds.length; i++) {
        if (batchCancelled.current) break
        const id = tripIds[i]
        setBatch((b) => ({ ...b, current: id }))
        const outcome = await runPrediction(id)
        if (outcome === 'success') ok++
        else if (outcome === 'error') failed++
        setBatch((b) => ({
          ...b,
          done: i + 1,
          succeeded: outcome === 'success' ? [...b.succeeded, id] : b.succeeded,
          failed: outcome === 'error' ? [...b.failed, id] : b.failed,
        }))
      }
      inBatch.current = false
      setBatch((b) => ({ ...b, running: false, current: null, finishedAt: Date.now(), stopped: batchCancelled.current }))
      const summary = `${ok} succeeded, ${failed} failed${batchCancelled.current ? ', stopped early' : ''}.`
      toast(failed ? 'error' : 'success', `Finished running ${label}`, summary, 6000)
    },
    [runPrediction],
  )

  const runAllTrips = useCallback(async () => {
    if (trips.status !== 'ready') return
    await runBatch(
      trips.trips.map((t) => t.trip_id),
      'all trips',
    )
  }, [trips, runBatch])

  const cancelBatch = useCallback(() => {
    batchCancelled.current = true
    setBatch((b) => {
      if (b.current) controllers.current.get(b.current)?.abort()
      return b
    })
  }, [])

  // ---- API address --------------------------------------------------------
  const setApiUrl = useCallback((url: string | null) => {
    saveApiBaseUrl(url)
    const next = resolveApiBaseUrl()
    for (const c of controllers.current.values()) c.abort()
    controllers.current.clear()
    setApiBaseUrl(next.url)
    setApiUrlState(next.url)
    setApiUrlSource(next.source)
    // Results from a different backend would be misleading; start clean.
    setPredictions({})
    setBatch(IDLE_BATCH)
    setTrips({ status: 'idle', trips: [] })
    setHealth({ status: 'checking' })
    setFocusIndex(null)
  }, [])

  // A remembered selection only counts if the live trip list still contains it.
  const selectedTrip = useMemo(
    () => trips.trips.find((t) => t.trip_id === storedTripId) ?? null,
    [trips.trips, storedTripId],
  )
  const selectedTripId = selectedTrip?.trip_id ?? null

  const isStubModel = health.data?.model?.toLowerCase() === 'stub'

  const value: SessionValue = {
    apiUrl,
    apiUrlSource,
    setApiUrl,
    health,
    healthChecking,
    refreshHealth: () => void refreshHealth(),
    isStubModel,
    trips,
    loadTrips: () => void loadTrips(),
    selectedTripId,
    selectTrip,
    selectedTrip,
    predictions,
    runPrediction,
    cancelPrediction,
    batch,
    runBatch,
    runAllTrips,
    cancelBatch,
    focusIndex,
    setFocusIndex,
  }

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSession(): SessionValue {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error('useSession must be used inside <SessionProvider>')
  return ctx
}

/** The selected trip's prediction record, if any. */
export function useSelectedPrediction(): PredictionRecord | undefined {
  const { predictions, selectedTripId } = useSession()
  return selectedTripId ? predictions[selectedTripId] : undefined
}
