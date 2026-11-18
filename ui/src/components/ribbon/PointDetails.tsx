import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { fmtCoord, fmtInt } from '@/lib/format'
import type { ArcRun } from '@/lib/sequence'
import { cn } from '@/lib/utils'

export interface PointInfo {
  index: number
  total: number
  /** [lon, lat] from the GPS layer, if the response included it. */
  point: [number, number] | null
  arc: number | null | undefined
  run: ArcRun | null
  /** 0-based positions of this arc in route_arcs. */
  routePos: number[] | undefined
  routeLength: number
}

/**
 * Everything the API response says about one GPS observation: where it is, which arc it was
 * matched to, the run of points sharing that arc, and where that arc sits in the route.
 */
export function PointDetails({
  info,
  onStep,
  onClose,
  dense = false,
}: {
  info: PointInfo
  onStep: (delta: number) => void
  onClose?: () => void
  dense?: boolean
}) {
  const { index, total, point, arc, run, routePos, routeLength } = info
  const matched = arc !== null && arc !== undefined
  const row = 'flex items-baseline justify-between gap-3'
  return (
    <div className={cn('text-[13px]', dense && 'text-[12.5px]')}>
      <div className="flex items-center justify-between gap-2">
        <p className="font-medium text-ink">
          GPS point <span className="num text-[17px] font-semibold">{fmtInt(index + 1)}</span>
          <span className="text-ink-faint"> of {fmtInt(total)}</span>
        </p>
        <div className="flex items-center gap-0.5">
          <button
            type="button"
            onClick={() => onStep(-1)}
            disabled={index === 0}
            aria-label="Previous point"
            className="grid size-6 place-items-center rounded text-ink-muted hover:bg-raised hover:text-ink disabled:opacity-40"
          >
            <ChevronLeft className="size-4" />
          </button>
          <button
            type="button"
            onClick={() => onStep(1)}
            disabled={index >= total - 1}
            aria-label="Next point"
            className="grid size-6 place-items-center rounded text-ink-muted hover:bg-raised hover:text-ink disabled:opacity-40"
          >
            <ChevronRight className="size-4" />
          </button>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Clear selected point"
              className="grid size-6 place-items-center rounded text-ink-muted hover:bg-raised hover:text-ink"
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>
      </div>

      <div
        className={cn(
          'mt-2 flex items-center gap-2 rounded-md border px-2.5 py-1.5',
          matched ? 'border-route/35 bg-route/[0.08]' : 'border-hairline bg-inset',
        )}
      >
        <span
          className={cn('size-2.5 shrink-0 rounded-full', matched ? 'bg-gps' : 'border-[1.5px] border-unmatched')}
          aria-hidden="true"
        />
        {matched ? (
          <span>
            Matched to arc <span className="num text-[16px] font-semibold text-route">{arc}</span>
          </span>
        ) : (
          <span className="text-ink-muted">Unmatched: the API returned no arc for this point</span>
        )}
      </div>

      <dl className="mt-2 space-y-1">
        {point && (
          <>
            <div className={row}>
              <dt className="text-ink-faint">Latitude</dt>
              <dd>{fmtCoord(point[1])}</dd>
            </div>
            <div className={row}>
              <dt className="text-ink-faint">Longitude</dt>
              <dd>{fmtCoord(point[0])}</dd>
            </div>
          </>
        )}
        {run && (
          <div className={row}>
            <dt className="text-ink-faint">{matched ? 'Run on this arc' : 'Unmatched stretch'}</dt>
            <dd>
              points {fmtInt(run.start + 1)}–{fmtInt(run.end + 1)}{' '}
              <span className="text-ink-faint">({fmtInt(run.length)})</span>
            </dd>
          </div>
        )}
        {matched && (
          <div className={row}>
            <dt className="text-ink-faint">In reconstructed route</dt>
            <dd>
              {routePos && routePos.length ? (
                <>
                  arc {routePos.map((p) => fmtInt(p + 1)).join(', ')}
                  <span className="text-ink-faint"> of {fmtInt(routeLength)}</span>
                </>
              ) : (
                <span className="text-ink-muted">not in route_arcs</span>
              )}
            </dd>
          </div>
        )}
      </dl>
    </div>
  )
}
