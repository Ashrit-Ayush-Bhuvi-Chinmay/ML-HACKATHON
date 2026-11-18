import type { TimingsMs } from '@/api/types'
import { fmtMs, stagesOf } from '@/lib/format'
import { cn } from '@/lib/utils'

const TONES = ['var(--ps-series-1)', 'color-mix(in oklab, var(--ps-series-1) 55%, var(--ps-surface))']

/**
 * Proportional strip of the server's stage timings (single hue, alternating shade, 2px gaps).
 * Only stages present in timings_ms are drawn; "Other" is total minus the listed stages.
 */
export function StageStrip({ timings, className, showLegend = true }: { timings: TimingsMs; className?: string; showLegend?: boolean }) {
  const { stages, total, unattributed } = stagesOf(timings)
  if (stages.length === 0) {
    return (
      <p className={cn('text-[12.5px] text-ink-muted', className)}>
        Server total <span className="num text-[15px] font-semibold text-ink">{fmtMs(total)}</span>. The API did not report per-stage times.
      </p>
    )
  }
  const parts = [
    ...stages.map((s) => ({ key: s.key, label: s.label, ms: s.ms, other: false })),
    ...(unattributed !== null ? [{ key: 'other', label: 'Other', ms: unattributed, other: true }] : []),
  ]
  const sum = parts.reduce((n, p) => n + p.ms, 0) || 1
  return (
    <div className={className}>
      <div className="flex h-3 gap-[2px] overflow-hidden rounded-[3px]" role="img" aria-label={`Server time ${fmtMs(total)} split by stage`}>
        {parts.map((p, i) => (
          <span
            key={p.key}
            title={`${p.label}: ${fmtMs(p.ms)}`}
            className="h-full min-w-[3px]"
            style={{ flexGrow: p.ms, flexBasis: 0, background: p.other ? 'var(--ps-ink-faint)' : TONES[i % 2] }}
          />
        ))}
      </div>
      {showLegend && (
        <ul className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[12px]">
          {parts.map((p, i) => (
            <li key={p.key} className="flex items-center gap-1.5">
              <span
                className="size-2 shrink-0 rounded-[2px]"
                style={{ background: p.other ? 'var(--ps-ink-faint)' : TONES[i % 2] }}
                aria-hidden="true"
              />
              <span className="truncate text-ink-muted">{p.label}</span>
              <span className="ml-auto text-ink">{fmtMs(p.ms)}</span>
              <span className="w-9 text-right text-ink-faint">{Math.round((p.ms / sum) * 100)}%</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
