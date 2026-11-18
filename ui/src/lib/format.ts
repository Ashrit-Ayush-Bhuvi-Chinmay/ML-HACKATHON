import type { TimingsMs } from '@/api/types'

export const NA = 'N/A'

const intFmt = new Intl.NumberFormat('en-GB')

export function fmtInt(v: number | null | undefined): string {
  return v === null || v === undefined || !Number.isFinite(v) ? NA : intFmt.format(Math.round(v))
}

/** Ratio in [0, 1] shown as a percentage. */
export function fmtPct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !Number.isFinite(v) ? NA : `${(v * 100).toFixed(digits)}%`
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return NA
  if (v >= 10_000) return `${(v / 1000).toFixed(1)} s`
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`
  if (v === 0) return '0 ms'
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ms`
}

export function fmtCoord(v: number): string {
  return v.toFixed(6)
}

export function fmtClock(ts: number): string {
  return new Date(ts).toLocaleTimeString('en-GB', { hour12: false })
}

export function fmtRelative(ts: number, now = Date.now()): string {
  const s = Math.max(0, Math.round((now - ts) / 1000))
  if (s < 5) return 'just now'
  if (s < 60) return `${s} s ago`
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min ago`
  return fmtClock(ts)
}

/** Contract stage order; any extra keys the backend adds are appended. */
export const STAGE_ORDER = [
  'load',
  'candidates',
  'features',
  'classifier',
  'hmm',
  'decode',
  'route',
  'continuity',
  'response',
] as const

export const STAGE_LABELS: Record<string, string> = {
  load: 'Load trip',
  candidates: 'Candidate arcs',
  features: 'Features',
  classifier: 'Classifier',
  hmm: 'HMM / Viterbi',
  decode: 'Decode (classifier only)',
  route: 'Build route',
  continuity: 'Check continuity',
  response: 'Build response',
}

export interface StageTiming {
  key: string
  label: string
  ms: number
}

export function stagesOf(t: TimingsMs): { stages: StageTiming[]; total: number | null; unattributed: number | null } {
  const keys = [...STAGE_ORDER.filter((k) => k in t), ...Object.keys(t).filter((k) => k !== 'total' && !(STAGE_ORDER as readonly string[]).includes(k))]
  const stages = keys.map((k) => ({ key: k, label: STAGE_LABELS[k] ?? k, ms: t[k] as number }))
  const sum = stages.reduce((n, s) => n + s.ms, 0)
  const total = typeof t.total === 'number' ? t.total : stages.length ? sum : null
  const unattributed = total !== null && stages.length > 0 && total - sum > 0.5 ? total - sum : null
  return { stages, total, unattributed }
}
