import type { PredictionRecord } from '@/state/session'

export type TripStatus = 'not-run' | 'running' | 'done' | 'failed'

/** Display status of a trip from its prediction record (a cancelled run counts as not run). */
export function tripStatus(rec?: PredictionRecord): TripStatus {
  if (!rec) return 'not-run'
  if (rec.status === 'running') return 'running'
  if (rec.status === 'error' && rec.error?.kind !== 'cancelled') return 'failed'
  return rec.result ? 'done' : 'not-run'
}
