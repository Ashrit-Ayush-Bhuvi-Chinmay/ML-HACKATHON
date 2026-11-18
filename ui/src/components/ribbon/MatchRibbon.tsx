import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { InfoTip } from '@/components/common/InfoTip'
import { fmtInt, fmtPct } from '@/lib/format'
import { adjacentGap, runsOf, statsOf, type ArcRun } from '@/lib/sequence'
import { cn } from '@/lib/utils'
import { useSession } from '@/state/session'

const TONE_A = 'var(--ps-route)'
const TONE_B = 'color-mix(in oklab, var(--ps-route) 48%, var(--ps-surface))'
const LENS = 48

type TonedRun = ArcRun & { tone: string | null }

function tone(runs: ArcRun[]): TonedRun[] {
  return runs.reduce<{ out: TonedRun[]; k: number }>(
    (acc, r) => {
      if (r.arc === null) acc.out.push({ ...r, tone: null })
      else acc.out.push({ ...r, tone: acc.k++ % 2 === 0 ? TONE_A : TONE_B })
      return acc
    },
    { out: [], k: 0 },
  ).out
}

/**
 * One cell per GPS observation, in time order. Consecutive points on the same arc form a run;
 * runs alternate between two shades so every arc change is visible. Hatched = unmatched (null).
 * With `lens`, a detail strip below shows the 48 points around the selected point one by one.
 */
export function MatchRibbon({
  arcs,
  height = 28,
  className,
  showAxis = true,
  lens = false,
}: {
  arcs: (number | null)[]
  height?: number
  className?: string
  showAxis?: boolean
  lens?: boolean
}) {
  const { focusIndex, setFocusIndex } = useSession()
  const [hover, setHover] = useState<number | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const n = arcs.length

  const runs = useMemo(() => runsOf(arcs), [arcs])
  const toned = useMemo(() => tone(runs), [runs])
  const stats = useMemo(() => statsOf(arcs, runs), [arcs, runs])
  const toneByStart = useMemo(() => new Map(toned.map((r) => [r.start, r.tone])), [toned])

  const indexAt = (e: PointerEvent<HTMLDivElement>) => {
    const rect = ref.current!.getBoundingClientRect()
    const x = Math.min(Math.max(e.clientX - rect.left, 0), rect.width - 0.01)
    return Math.floor((x / rect.width) * n)
  }

  const onKey = (e: KeyboardEvent) => {
    if (!n) return
    const step = e.shiftKey ? 10 : 1
    const cur = focusIndex ?? -1
    if (e.key === 'ArrowRight') setFocusIndex(Math.min(n - 1, cur + step))
    else if (e.key === 'ArrowLeft') setFocusIndex(Math.max(0, cur === -1 ? 0 : cur - step))
    else if (e.key === 'Home') setFocusIndex(0)
    else if (e.key === 'End') setFocusIndex(n - 1)
    else if (e.key === 'Escape') setFocusIndex(null)
    else return
    e.preventDefault()
  }

  if (n === 0) {
    return <p className="text-[13px] text-ink-muted">The response contains no predicted arcs to plot.</p>
  }

  const allNull = stats.matched === 0
  const readout = hover ?? focusIndex
  const readoutArc = readout !== null ? arcs[readout] : undefined
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.min(n - 1, Math.round(f * (n - 1))))
  const prevGap = adjacentGap(runs, focusIndex ?? n, -1)
  const nextGap = adjacentGap(runs, focusIndex ?? -1, 1)

  const lensStart = focusIndex === null ? 0 : Math.max(0, Math.min(n - LENS, focusIndex - LENS / 2))
  const lensEnd = Math.min(n - 1, lensStart + LENS - 1)

  return (
    <div className={cn('select-none', className)}>
      {/* Summary + navigation */}
      <div className="mb-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5 text-[12.5px]">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span className="flex items-center gap-1.5 text-ink">
            <span className="flex h-2.5 w-5 overflow-hidden rounded-[2px]" aria-hidden="true">
              <span className="flex-1" style={{ background: TONE_A }} />
              <span className="flex-1" style={{ background: TONE_B }} />
            </span>
            <span className="num text-[14px] font-semibold">{fmtInt(stats.matched)}</span>
            <span className="text-ink-muted">matched ({fmtPct(stats.matched / n)})</span>
          </span>
          <span className="flex items-center gap-1.5 text-ink">
            <span className="hatch h-2.5 w-5 rounded-[2px] border border-hairline" aria-hidden="true" />
            <span className="num text-[14px] font-semibold">{fmtInt(stats.unmatched)}</span>
            <span className="text-ink-muted">
              unmatched in {fmtInt(stats.unmatchedRuns)} {stats.unmatchedRuns === 1 ? 'stretch' : 'stretches'}
            </span>
          </span>
          <InfoTip label="How to read the ribbon">
            Each thin slice is one GPS observation, left to right in time order. A solid block is a run of consecutive points matched to
            the same arc; the shade flips each time the arc changes. Hatched slices are points the API left unmatched (null).
          </InfoTip>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => prevGap !== null && setFocusIndex(prevGap)}
            disabled={prevGap === null}
            className="flex h-7 items-center gap-1 rounded-md border border-hairline px-2 text-ink-muted transition-colors hover:bg-raised hover:text-ink disabled:opacity-40"
          >
            <ChevronLeft className="size-3.5" /> Previous gap
          </button>
          <button
            type="button"
            onClick={() => nextGap !== null && setFocusIndex(nextGap)}
            disabled={nextGap === null}
            className="flex h-7 items-center gap-1 rounded-md border border-hairline px-2 text-ink-muted transition-colors hover:bg-raised hover:text-ink disabled:opacity-40"
          >
            Next gap <ChevronRight className="size-3.5" />
          </button>
        </div>
      </div>

      {/* Ribbon */}
      <div
        ref={ref}
        role="slider"
        tabIndex={0}
        aria-label="GPS point sequence. Arrow keys move the selected point, Shift moves by ten."
        aria-valuemin={1}
        aria-valuemax={n}
        aria-valuenow={focusIndex !== null ? focusIndex + 1 : undefined}
        aria-valuetext={focusIndex !== null ? `Point ${focusIndex + 1}, arc ${arcs[focusIndex] ?? 'unmatched'}` : 'No point selected'}
        onKeyDown={onKey}
        onPointerMove={(e) => setHover(indexAt(e))}
        onPointerLeave={() => setHover(null)}
        onPointerDown={(e) => setFocusIndex(indexAt(e))}
        className="relative cursor-crosshair overflow-hidden rounded-[4px] border border-hairline bg-inset outline-none focus-visible:ring-2 focus-visible:ring-route"
        style={{ height }}
      >
        {allNull ? (
          <div className="hatch absolute inset-0" />
        ) : (
          <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${n} 1`} preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <pattern id="ribbon-hatch" width="6" height="1" patternUnits="userSpaceOnUse" patternTransform="skewX(-30)">
                <rect width="2" height="1" fill="var(--ps-ink-faint)" opacity="0.7" />
              </pattern>
            </defs>
            {toned.map((r) => (
              <rect key={r.start} x={r.start} width={r.length} y={0} height={1} fill={r.tone ?? 'url(#ribbon-hatch)'} shapeRendering="crispEdges" />
            ))}
          </svg>
        )}
        {lens && focusIndex !== null && (
          <span
            className="pointer-events-none absolute inset-y-0 border-x border-white/70 bg-white/10"
            style={{ left: `${(lensStart / n) * 100}%`, width: `${((lensEnd - lensStart + 1) / n) * 100}%` }}
            aria-hidden="true"
          />
        )}
        {hover !== null && <span className="pointer-events-none absolute inset-y-0 w-px bg-ink/70" style={{ left: `${((hover + 0.5) / n) * 100}%` }} />}
        {focusIndex !== null && focusIndex < n && (
          <span
            className="pointer-events-none absolute inset-y-[-2px] w-[3px] -translate-x-1/2 rounded-full bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.6)]"
            style={{ left: `${((focusIndex + 0.5) / n) * 100}%` }}
          />
        )}
      </div>

      <div className="mt-1 flex items-start justify-between gap-3">
        {showAxis ? (
          <div className="relative h-4 flex-1 text-[11px] text-ink-faint" aria-hidden="true">
            {ticks.map((t, i) => (
              <span
                key={i}
                className={cn('absolute', i === 0 ? 'left-0' : i === ticks.length - 1 ? 'right-0' : '-translate-x-1/2')}
                style={i > 0 && i < ticks.length - 1 ? { left: `${((t + 0.5) / n) * 100}%` } : undefined}
              >
                {fmtInt(t + 1)}
              </span>
            ))}
          </div>
        ) : (
          <span />
        )}
      </div>
      <p className="mt-0.5 min-h-[18px] text-[12.5px] text-ink" aria-live="polite">
        {readout !== null ? (
          <>
            Point <span className="num text-[14px] font-semibold">{fmtInt(readout + 1)}</span>
            <span className="text-ink-faint"> of {fmtInt(n)}</span>
            <span className="ml-2 text-ink-muted">{readoutArc === null ? 'unmatched' : 'arc'}</span>{' '}
            {readoutArc !== null && <span className="num text-[14px] font-semibold text-route">{readoutArc}</span>}
            {hover === null && <span className="ml-2 text-ink-faint">Arrow keys step through points.</span>}
          </>
        ) : (
          <span className="text-ink-faint">Hover the ribbon to read a point; click it to locate the point on the map.</span>
        )}
      </p>

      {lens && (
        <div className="mt-2 rounded-lg border border-hairline bg-inset p-3">
          <div className="mb-2 flex items-baseline justify-between gap-2 text-[12.5px]">
            <span className="font-medium text-ink">Point by point</span>
            <span className="text-ink-faint">
              {focusIndex === null ? `Points 1–${fmtInt(lensEnd + 1)}` : `Points ${fmtInt(lensStart + 1)}–${fmtInt(lensEnd + 1)}`}
            </span>
          </div>
          <div className="grid gap-[3px]" style={{ gridTemplateColumns: `repeat(${lensEnd - lensStart + 1}, minmax(0, 1fr))` }}>
            {Array.from({ length: lensEnd - lensStart + 1 }, (_, k) => {
              const i = lensStart + k
              const a = arcs[i]
              const runStart = i === 0 || arcs[i - 1] !== a
              return (
                <div key={i} className="min-w-0">
                  <span className={cn('block h-4 truncate text-[10.5px] leading-4', a === null ? 'text-ink-faint' : 'text-route')}>
                    {runStart ? (a === null ? 'gap' : a) : ''}
                  </span>
                  <button
                    type="button"
                    onClick={() => setFocusIndex(i)}
                    aria-label={`Point ${i + 1}, ${a === null ? 'unmatched' : `arc ${a}`}`}
                    aria-pressed={focusIndex === i}
                    className={cn(
                      'block h-7 w-full rounded-[3px]',
                      a === null && 'hatch border border-hairline',
                      focusIndex === i && 'ring-2 ring-white ring-offset-1 ring-offset-inset',
                    )}
                    style={a === null ? undefined : { background: toneByStart.get(runsStartOf(runs, i)) ?? TONE_A }}
                  />
                </div>
              )
            })}
          </div>
          <p className="mt-2 text-[12px] text-ink-faint">
            Labels mark where a new arc starts. Select a cell to inspect that point.
          </p>
        </div>
      )}
      {allNull && <p className="mt-1 text-[12.5px] text-ink-muted">No GPS point was matched to an arc in this response.</p>}
    </div>
  )
}

/** Start index of the run containing point i (runs are contiguous and ordered). */
function runsStartOf(runs: ArcRun[], i: number): number {
  let lo = 0
  let hi = runs.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (i < runs[mid].start) hi = mid - 1
    else if (i > runs[mid].end) lo = mid + 1
    else return runs[mid].start
  }
  return i
}
