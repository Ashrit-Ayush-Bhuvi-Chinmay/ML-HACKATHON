import type { PredictionMetrics } from '@/api/types'
import { InfoTip } from '@/components/common/InfoTip'
import { fmtInt, fmtPct, NA } from '@/lib/format'
import { cn } from '@/lib/utils'

interface MetricDef {
  key: Exclude<keyof PredictionMetrics, 'note'>
  label: string
  hint: string
  ratio: boolean
  /** Extra explanation shown under a null value, instead of the generic line. */
  missingText?: (m: PredictionMetrics) => string
}

export const METRIC_DEFS: MetricDef[] = [
  {
    key: 'top1_accuracy',
    label: 'Top-1 accuracy',
    hint: 'Share of GPS points matched to the right road segment. For each observation, the highest-scoring candidate arc is compared with the reference arc on the server.',
    ratio: true,
  },
  {
    key: 'sequence_f1',
    label: 'Sequence F1',
    hint: 'Did we recover the journey? Order- and lap-aware route score, so a wrong order or a missed lap lowers it. This is the main route metric.',
    ratio: true,
  },
  {
    key: 'coverage_f1',
    label: 'Coverage F1',
    hint: 'Did we cover the right roads? Length-weighted and order-independent, so it credits the right roads even when their order is wrong.',
    ratio: true,
  },
  {
    key: 'route_continuity',
    label: 'Route continuity',
    hint: 'Share of segment changes a car could physically make. N/A means there was nothing to judge (for example no segment changes).',
    ratio: true,
    missingText: (m) =>
      typeof m.n_transitions === 'number'
        ? `Nothing to judge (${fmtInt(m.n_transitions)} transitions)`
        : 'Nothing to judge',
  },
  {
    key: 'route_mismatch',
    label: 'Route mismatch',
    hint: 'Wrongly added plus missed route length, as a fraction of the true route. Lower is better.',
    ratio: true,
  },
  {
    key: 'n_points_scored',
    label: 'Points scored',
    hint: 'GPS points with a reliable reference label. A value of 0 is a real zero returned by the API, not missing data.',
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
        const missing = v === null || v === undefined
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
            {missing && (
              <p className="mt-1.5 text-[11.5px] leading-tight text-ink-faint">
                {m.missingText ? m.missingText(metrics) : 'Not returned by the API'}
              </p>
            )}
          </div>
        )
      })}
    </dl>
  )
}

export function allMetricsNull(m: PredictionMetrics): boolean {
  return METRIC_DEFS.every((d) => m[d.key] === null || (d.key === 'n_points_scored' && m[d.key] === 0))
}
