/**
 * HTTP client for the PathSnap FastAPI backend.
 *
 * - One axios instance; its baseURL can change at runtime (API address setting).
 * - Every call is recorded in a small in-memory request log so the evaluator can
 *   match UI requests against the API console on Laptop 1.
 * - All failures are normalised into ApiError with a kind the UI can explain.
 */
import axios, { AxiosError, type AxiosRequestConfig } from 'axios'
import { DEFAULT_TIMEOUT_MS, resolveApiBaseUrl } from '@/config'
import { InvalidResponseError } from './validate'

export type ApiErrorKind =
  | 'network' // backend unreachable (wrong IP, firewall, API not running, CORS)
  | 'timeout'
  | 'not_found' // 404, e.g. unknown trip_id
  | 'server' // 5xx
  | 'client' // other 4xx
  | 'invalid_response'
  | 'cancelled'

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status?: number
  readonly detail?: string
  constructor(kind: ApiErrorKind, message: string, opts: { status?: number; detail?: string } = {}) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = opts.status
    this.detail = opts.detail
  }
}

export interface RequestLogEntry {
  id: number
  method: string
  path: string
  startedAt: number
  durationMs: number | null
  status: number | null
  outcome: 'pending' | 'ok' | 'error'
  message?: string
  /** Short context for the log line, e.g. "trip 00000007". */
  note?: string
  /** Base URL the request was sent to. */
  baseUrl: string
}

type LogListener = (entries: RequestLogEntry[]) => void

const MAX_LOG = 100
let logEntries: RequestLogEntry[] = []
let nextId = 1
const listeners = new Set<LogListener>()

function emit() {
  for (const l of listeners) l(logEntries)
}

function upsertLog(entry: RequestLogEntry) {
  const i = logEntries.findIndex((e) => e.id === entry.id)
  logEntries = i === -1 ? [entry, ...logEntries].slice(0, MAX_LOG) : logEntries.map((e) => (e.id === entry.id ? entry : e))
  emit()
}

export function subscribeRequestLog(listener: LogListener): () => void {
  listeners.add(listener)
  listener(logEntries)
  return () => listeners.delete(listener)
}

export function clearRequestLog() {
  logEntries = []
  emit()
}

const http = axios.create({
  baseURL: resolveApiBaseUrl().url,
  timeout: DEFAULT_TIMEOUT_MS,
  headers: { Accept: 'application/json' },
})

export function setApiBaseUrl(url: string) {
  http.defaults.baseURL = url
}

export function getApiBaseUrl(): string {
  return http.defaults.baseURL ?? ''
}

function detailOf(data: unknown): string | undefined {
  if (data && typeof data === 'object' && 'detail' in data) {
    const d = (data as { detail: unknown }).detail
    return typeof d === 'string' ? d : JSON.stringify(d)
  }
  return undefined
}

function toApiError(err: unknown, path: string): ApiError {
  if (err instanceof ApiError) return err
  if (err instanceof InvalidResponseError) return new ApiError('invalid_response', err.message)
  if (axios.isCancel(err)) return new ApiError('cancelled', 'Request cancelled.')
  if (err instanceof AxiosError) {
    if (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT') {
      return new ApiError('timeout', `The API did not answer ${path} in time.`)
    }
    if (err.response) {
      const status = err.response.status
      const detail = detailOf(err.response.data)
      if (status === 404) return new ApiError('not_found', detail ?? 'Not found.', { status, detail })
      if (status >= 500) return new ApiError('server', detail ?? `Server error ${status}.`, { status, detail })
      return new ApiError('client', detail ?? `Request rejected (${status}).`, { status, detail })
    }
    return new ApiError(
      'network',
      `Cannot reach the API at ${getApiBaseUrl()}. Check that it is running, the address and port are right, the port is open in Laptop 1's firewall and CORS is enabled.`,
    )
  }
  return new ApiError('network', err instanceof Error ? err.message : 'Unknown error.')
}

/** Performs a request, records it in the log, validates the body. */
export async function request<T>(
  config: AxiosRequestConfig & { url: string; logNote?: string },
  validate: (data: unknown) => T,
): Promise<T> {
  const { logNote, ...axiosConfig } = config
  const method = (config.method ?? 'GET').toUpperCase()
  const entry: RequestLogEntry = {
    id: nextId++,
    method,
    path: config.url,
    note: logNote,
    baseUrl: getApiBaseUrl(),
    startedAt: Date.now(),
    durationMs: null,
    status: null,
    outcome: 'pending',
  }
  upsertLog(entry)
  const t0 = performance.now()
  try {
    const res = await http.request({ ...axiosConfig, responseType: 'json' })
    const value = validate(res.data)
    upsertLog({ ...entry, durationMs: performance.now() - t0, status: res.status, outcome: 'ok' })
    return value
  } catch (err) {
    const apiErr = toApiError(err, config.url)
    upsertLog({
      ...entry,
      durationMs: performance.now() - t0,
      status: apiErr.status ?? (err instanceof AxiosError ? (err.response?.status ?? null) : null),
      outcome: 'error',
      message: apiErr.message,
    })
    throw apiErr
  }
}
