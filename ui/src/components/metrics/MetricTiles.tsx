import type { PredictionMetrics } from '@/api/types'
import { InfoTip } from '@/components/common/InfoTip'
import { fmtInt, fmtPct, NA } from '@/lib/format'
import { cn } from '@/lib/utils'

export const METRIC_DEFS: { key: keyof PredictionMetrics; label: string; hint: string; ratio: boolean }[] = [
  {
    key: 'top1_accuracy',
    label: 'Top-1 accuracy',
    hint: 'Observation level: for each GPS point, the highest-scoring candidate arc is compared with the reference arc. Computed on the API server; the reference route never reaches this UI.',
    ratio: true,
  },
  {
    key: 'route_f1',
    label: 'Route F1',
    hint: 'Route-level F1 score computed by the API’s score_trip() against the reference route. See the backend team’s implementation for the exact definition.',
    ratio: true,
  },
  {
    key: 'route_continuity',
    label: 'Route continuity',
    hint: 'How connected the predicted route is, as computed by the API’s score_trip(). See the backend implementation for the exact definition.',
    ratio: true,
  },
  {
    key: 'route_precision',
    label: 'Route precision',
    hint: 'Route-level precision computed by the API’s score_trip() against the reference route.',
    ratio: true,
  },
  {
    key: 'route_recall',
    label: 'Route recall',
    hint: 'Route-level recall computed by the API’s score_trip() against the reference route.',
    ratio: true,
  },
  {
    key: 'n_points_scored',
    label: 'Points scored',
    hint: 'How many GPS points the API scored. A value of 0 is a real zero returned by the API, not missing data.',
    ratio: false,
  },
]

function Meter({ value }: { value: number }) {
  const w = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="mt-2.5 h-[4px] overflow-hidden rounded-full bg-hairline/80" aria-hidden="true">
      <div className="h-full rounded-full bg-route" style={{ width: `${w}%` }} />
    </div>
  )
}

/**
 * Metric values exactly as returned by the API.
 * null -> "N/A" on a dashed, recessed tile; 0 -> a real "0", drawn like any other value.
 */
export function MetricTiles({
  metrics,
  size = 'sm',
  className,
}: {
  metrics: PredictionMetrics
  size?: 'sm' | 'lg'
  className?: string
}) {
  return (
    <dl className={cn('grid gap-2', size === 'lg' ? 'grid-cols-2 lg:grid-cols-3' : 'grid-cols-2', className)}>
      {METRIC_DEFS.map((m) => {
        const v = metrics[m.key]
        const missing = v === null
        return (
          <div
            key={m.key}
            className={cn(
              'rounded-lg border',
              size === 'lg' ? 'px-4 py-3.5' : 'px-3 py-2.5',
              missing ? 'border-dashed border-hairline bg-transparent' : 'border-hairline bg-raised/45',
            )}
          >
            <dt className="flex items-center gap-1 text-[12.5px] text-ink-muted">
              {m.label}
              <InfoTip label={`About ${m.label}`}>{m.hint}</InfoTip>
            </dt>
            <dd
              className={cn(
                'num leading-none font-semibold',
                size === 'lg' ? 'mt-2 text-[36px]' : 'mt-1.5 text-[25px]',
                missing ? 'text-ink-faint' : 'text-ink',
              )}
            >
              {missing ? NA : m.ratio ? fmtPct(v) : fmtInt(v)}
            </dd>
            {!missing && m.ratio && <Meter value={v} />}
            {missing && <p className="mt-1.5 text-[11.5px] leading-tight text-ink-faint">Not returned by the API</p>}
          </div>
        )
      })}
    </dl>
  )
}

export function allMetricsNull(m: PredictionMetrics): boolean {
  return METRIC_DEFS.every((d) => m[d.key] === null || (d.key === 'n_points_scored' && m[d.key] === 0))
}
