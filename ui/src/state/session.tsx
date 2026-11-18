/**
 * Session state shared by every page: backend health, trip list, the selected trip,
 * prediction results per trip and the GPS point currently in focus (map <-> ribbon).
 *
 * Everything shown in the UI comes from these API responses. Nothing is cached across
 * page reloads except the selected trip id and the API address.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ApiError, setApiBaseUrl } from '@/api/client'
import { getHealth, getModels, getTrips, postPredict } from '@/api/services'
import type { HealthResponse, ModelInfo, PredictOptions, PredictResponse, TripSummary } from '@/api/types'
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
  /** Model name reported by the API for this prediction. */
  model: string | null
}

/** Identity of one stored run of a trip: model + decoder, as reported by the API. */
export function runKeyOf(result: PredictionResult): string {
  const model = result.response.model ?? result.model ?? 'unknown'
  return `${model}|${result.response.hmm === false ? 'classifier' : 'hmm'}`
}

/** Human label for a stored run, e.g. "distance_gaussian · no HMM". */
export function runLabelOf(result: PredictionResult): string {
  const model = result.response.model ?? result.model ?? 'unknown'
  return `${model} · ${result.response.hmm === false ? 'no HMM' : 'HMM'}`
}

/** Does a stored run correspond to the requested model + decoder? "default" means the deployed model. */
export function matchesRunOptions(
  result: PredictionResult,
  options: PredictOptions,
  deployedModel?: string | null,
): boolean {
  const requested = options.model && options.model !== 'default' ? options.model : (deployedModel ?? null)
  const wantsHmm = options.hmm !== false
  const modelOk = !requested || (result.response.model ?? result.model) === requested
  return modelOk && (result.response.hmm !== false) === wantsHmm
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
  /** Stop was requested; the batch ends once the current trip finishes. */
  stopping: boolean
}

const NO_KEYS: string[] = []

const IDLE_BATCH: BatchState = {  running: false,
  label: '',
  queue: [],
  done: 0,
  current: null,
  succeeded: [],
  failed: [],
  startedAt: null,
  finishedAt: null,
  stopped: false,
  stopping: false,
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

  /** Saved models from GET /models (leaderboard + picker). */
  models: ModelInfo[]
  modelsStatus: 'idle' | 'loading' | 'ready' | 'error'
  loadModels: () => void

  selectedTripId: string | null
  selectTrip: (id: string | null) => void
  selectedTrip: TripSummary | null

  predictions: Record<string, PredictionRecord>
  runPrediction: (tripId: string, options?: PredictOptions, reuse?: boolean, notify?: boolean) => Promise<RunOutcome>
  cancelPrediction: (tripId: string) => void
  /** Every stored successful run of a trip, newest first, keyed by model + decoder. */
  runsForTrip: (tripId: string) => PredictionResult[]
  /** The stored run for a given model + decoder, if it exists. */
  findStoredRun: (tripId: string, options?: PredictOptions) => PredictionResult | null
  /** Run (or reuse) a model without replacing the current result; used for map overlays. */
  fetchRun: (tripId: string, options?: PredictOptions) => Promise<PredictionResult>
  /** Run keys selected as translucent overlays, per trip. */
  overlayKeysFor: (tripId: string) => string[]
  setOverlayKeysFor: (tripId: string, keys: string[]) => void

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
  const [models, setModels] = useState<ModelInfo[]>([])
  const [modelsStatus, setModelsStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')
  const [storedTripId, setSelectedTripId] = useState<string | null>(readSelected)
  const [predictions, setPredictions] = useState<Record<string, PredictionRecord>>({})
  const [runs, setRuns] = useState<Record<string, PredictionResult[]>>({})
  const [overlayKeys, setOverlayKeys] = useState<Record<string, string[]>>({})
  const [batch, setBatch] = useState<BatchState>(IDLE_BATCH)
  const [focusIndex, setFocusIndex] = useState<number | null>(null)

  const controllers = useRef(new Map<string, AbortController>())
  const batchCancelled = useRef(false)
  const inBatch = useRef(false)
  /** Bumped when the API address changes; work started under an older epoch is discarded. */
  const epoch = useRef(0)
  const healthRef = useRef(health)
  const tripsStatusRef = useRef(trips.status)
  const runsRef = useRef(runs)
  useEffect(() => {
    healthRef.current = health
    tripsStatusRef.current = trips.status
  }, [health, trips.status])
  useEffect(() => {
    runsRef.current = runs
  }, [runs])

  // ---- models -------------------------------------------------------------
  const loadModels = useCallback(async () => {
    const myEpoch = epoch.current
    setModelsStatus('loading')
    try {
      const res = await getModels()
      if (myEpoch !== epoch.current) return
      setModels(res.models)
      setModelsStatus('ready')
    } catch {
      if (myEpoch !== epoch.current) return
      setModelsStatus('error')
    }
  }, [])

  useEffect(() => {
    if (health.status === 'online' && (modelsStatus === 'idle' || modelsStatus === 'error')) {
      void loadModels()
    }
  }, [health.status, modelsStatus, loadModels])

  // ---- trips --------------------------------------------------------------
  const loadTrips = useCallback(async () => {
    const myEpoch = epoch.current
    tripsStatusRef.current = 'loading'
    setTrips((t) => ({ status: 'loading', trips: t.trips }))
    try {
      const res = await getTrips()
      if (myEpoch !== epoch.current) return
      setTrips({ status: 'ready', trips: res.trips, loadedAt: Date.now() })
    } catch (e) {
      if (myEpoch !== epoch.current) return
      setTrips((t) => ({ status: 'error', trips: t.trips, error: e as ApiError }))
    }
  }, [])

  // ---- health -------------------------------------------------------------
  const refreshHealth = useCallback(async () => {
    const myEpoch = epoch.current
    setHealthChecking(true)
    const t0 = performance.now()
    try {
      const data = await getHealth()
      if (myEpoch !== epoch.current) return
      setHealth({ status: 'online', data, checkedAt: Date.now(), latencyMs: performance.now() - t0 })
      // Load trips as soon as the backend answers, and again after an outage or a failed load.
      const ts = tripsStatusRef.current
      if (ts === 'idle' || ts === 'error') void loadTrips()
    } catch (e) {
      if (myEpoch !== epoch.current) return
      setHealth({ status: 'offline', error: e as ApiError, checkedAt: Date.now() })
    } finally {
      if (myEpoch === epoch.current) setHealthChecking(false)
    }
  }, [loadTrips])

  // Polling: first check on the next tick, then every HEALTH_POLL_MS. Restarts when the API address changes.
  useEffect(() => {
    const first = window.setTimeout(() => void refreshHealth(), 0)
    const id = window.setInterval(() => void refreshHealth(), HEALTH_POLL_MS)
    return () => {
      window.clearTimeout(first)
      window.clearInterval(id)
    }
  }, [refreshHealth, apiUrl])

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
  const findStoredRun = useCallback((tripId: string, options: PredictOptions = {}) => {
    const h = healthRef.current
    const deployed = h.status === 'online' ? h.data.model : null
    return (runsRef.current[tripId] ?? []).find((result) => matchesRunOptions(result, options, deployed)) ?? null
  }, [])

  /** Run (or reuse) a model in the background; the result is stored but does not replace the current one. */
  const fetchRun = useCallback(
    async (tripId: string, options: PredictOptions = {}): Promise<PredictionResult> => {
      const stored = findStoredRun(tripId, options)
      if (stored) return stored
      const t0 = performance.now()
      const { value, warnings } = await postPredict(tripId, options)
      const h = healthRef.current
      const result: PredictionResult = {
        response: value,
        warnings,
        receivedAt: Date.now(),
        roundTripMs: performance.now() - t0,
        model: value.model ?? (h.status === 'online' ? h.data.model : null),
      }
      const key = runKeyOf(result)
      const previous = runsRef.current[tripId] ?? []
      const nextRuns = { ...runsRef.current, [tripId]: [result, ...previous.filter((r) => runKeyOf(r) !== key)] }
      runsRef.current = nextRuns
      setRuns(nextRuns)
      return result
    },
    [findStoredRun],
  )

  const runPrediction = useCallback(async (tripId: string, options: PredictOptions = {}, reuse = false, notify = true): Promise<RunOutcome> => {
    if (reuse) {
      const stored = findStoredRun(tripId, options)
      if (stored) {
        setPredictions((p) => ({ ...p, [tripId]: { status: 'success', startedAt: Date.now(), result: stored } }))
        return 'success'
      }
    }

    controllers.current.get(tripId)?.abort()
    const myEpoch = epoch.current
    const controller = new AbortController()
    controllers.current.set(tripId, controller)
    const startedAt = Date.now()
    setPredictions((p) => ({ ...p, [tripId]: { ...p[tripId], status: 'running', startedAt, error: undefined } }))
    const t0 = performance.now()
    // A newer run of the same trip, or an API address change, makes this run stale: drop its outcome.
    const stale = () => myEpoch !== epoch.current || controllers.current.get(tripId) !== controller
    try {
      const { value, warnings } = await postPredict(tripId, options, controller.signal)
      if (stale()) return 'cancelled'
      const h = healthRef.current
      const result: PredictionResult = {
        response: value,
        warnings,
        receivedAt: Date.now(),
        roundTripMs: performance.now() - t0,
        model: value.model ?? (h.status === 'online' ? h.data.model : null),
      }
      setPredictions((p) => ({ ...p, [tripId]: { status: 'success', startedAt, result } }))
      // Keep every run of this trip, newest first, one entry per model + decoder.
      const key = runKeyOf(result)
      const previous = runsRef.current[tripId] ?? []
      const nextRuns = {
        ...runsRef.current,
        [tripId]: [result, ...previous.filter((r) => runKeyOf(r) !== key)],
      }
      runsRef.current = nextRuns
      setRuns(nextRuns)
      if (!inBatch.current && notify) {
        toast('success', `Prediction received for trip ${tripId}`, `${runLabelOf(result)} · server ${fmtMs(value.timings_ms.total)}, round trip ${fmtMs(result.roundTripMs)}.`)
      }
      return 'success'
    } catch (e) {
      const err = e as ApiError
      if (stale()) return 'cancelled'
      setPredictions((p) => ({ ...p, [tripId]: { ...p[tripId], status: 'error', startedAt, error: err } }))
      if (!inBatch.current && notify) {
        if (err.kind === 'cancelled') toast('info', `Prediction cancelled for trip ${tripId}`)
        else toast('error', `Prediction failed for trip ${tripId}`, err.message)
      }
      return err.kind === 'cancelled' ? 'cancelled' : 'error'
    } finally {
      if (controllers.current.get(tripId) === controller) controllers.current.delete(tripId)
    }
  }, [findStoredRun])

  const cancelPrediction = useCallback((tripId: string) => {
    controllers.current.get(tripId)?.abort()
  }, [])

  const runBatch = useCallback(
    async (tripIds: string[], label: string) => {
      if (tripIds.length === 0 || inBatch.current) return
      const myEpoch = epoch.current
      const live = () => myEpoch === epoch.current
      batchCancelled.current = false
      inBatch.current = true
      setBatch({ ...IDLE_BATCH, running: true, label, queue: tripIds, startedAt: Date.now() })
      let ok = 0
      let failed = 0
      try {
        for (let i = 0; i < tripIds.length; i++) {
          if (batchCancelled.current || !live()) break
          const id = tripIds[i]
          setBatch((b) => ({ ...b, current: id }))
          // A failed trip is recorded and the batch moves on to the next one.
          const outcome = await runPrediction(id)
          if (!live()) break
          if (outcome === 'success') ok++
          else if (outcome === 'error') failed++
          setBatch((b) => ({
            ...b,
            done: i + 1,
            succeeded: outcome === 'success' ? [...b.succeeded, id] : b.succeeded,
            failed: outcome === 'error' ? [...b.failed, id] : b.failed,
          }))
        }
      } finally {
        if (live()) inBatch.current = false
      }
      if (!live()) return
      const stopped = batchCancelled.current
      setBatch((b) => ({ ...b, running: false, current: null, finishedAt: Date.now(), stopped, stopping: false }))
      const summary = `${ok} succeeded, ${failed} failed${stopped ? ', stopped early' : ''}.`
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

  /** Stops the batch after the trip currently being predicted; that request is not aborted. */
  const cancelBatch = useCallback(() => {
    batchCancelled.current = true
    setBatch((b) => (b.running ? { ...b, stopping: true } : b))
  }, [])

  // ---- API address --------------------------------------------------------
  const setApiUrl = useCallback((url: string | null) => {
    saveApiBaseUrl(url)
    const next = resolveApiBaseUrl()
    // Invalidate all in-flight work from the previous backend, including a running batch.
    epoch.current++
    batchCancelled.current = true
    inBatch.current = false
    tripsStatusRef.current = 'idle'
    for (const c of controllers.current.values()) c.abort()
    controllers.current.clear()
    setApiBaseUrl(next.url)
    setApiUrlState(next.url)
    setApiUrlSource(next.source)
    // Results from a different backend would be misleading; start clean.
    setPredictions({})
    setRuns({})
    setOverlayKeys({})
    setBatch(IDLE_BATCH)
    setTrips({ status: 'idle', trips: [] })
    setModels([])
    setModelsStatus('idle')
    setHealth({ status: 'checking' })
    setHealthChecking(false)
    setFocusIndex(null)
  }, [])

  // A remembered selection only counts if the live trip list still contains it.
  const selectedTrip = useMemo(
    () => trips.trips.find((t) => t.trip_id === storedTripId) ?? null,
    [trips.trips, storedTripId],
  )
  const selectedTripId = selectedTrip?.trip_id ?? null

  const runsForTrip = useCallback((tripId: string) => runs[tripId] ?? [], [runs])
  const overlayKeysFor = useCallback((tripId: string) => overlayKeys[tripId] ?? NO_KEYS, [overlayKeys])
  const setOverlayKeysFor = useCallback((tripId: string, keys: string[]) => {
    setOverlayKeys((all) => ({ ...all, [tripId]: keys }))
  }, [])

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
    models,
    modelsStatus,
    loadModels: () => void loadModels(),
    selectedTripId,
    selectTrip,
    selectedTrip,
    predictions,
    runPrediction,
    cancelPrediction,
    runsForTrip,
    findStoredRun,
    fetchRun,
    overlayKeysFor,
    setOverlayKeysFor,
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
