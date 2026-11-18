import { useMemo, useState } from 'react'
import { ArrowDownUp, CheckCircle2, CircleDashed, CircleX, FolderSearch, Loader2, RotateCw, Search, X } from 'lucide-react'
import { ApiErrorNotice } from '@/components/common/ApiErrorNotice'
import { EmptyState } from '@/components/common/EmptyState'
import { Segmented } from '@/components/common/Segmented'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { fmtInt, fmtPct } from '@/lib/format'
import { cn } from '@/lib/utils'
import { useSession, type PredictionRecord } from '@/state/session'

import { tripStatus } from '@/lib/tripStatus'

export { tripStatus, type TripStatus } from '@/lib/tripStatus'

function RowStatus({ rec }: { rec?: PredictionRecord }) {
  const s = tripStatus(rec)
  if (s === 'running')
    return (
      <span className="flex items-center gap-1 text-route">
        <Loader2 className="size-3.5 animate-spin" /> Predicting
      </span>
    )
  if (s === 'failed')
    return (
      <span className="flex items-center gap-1 text-err">
        <CircleX className="size-3.5" /> Failed
      </span>
    )
  if (s === 'done') {
    const top1 = rec!.result!.response.metrics.top1_accuracy
    return (
      <span className="flex items-center gap-1 text-ink-muted">
        <CheckCircle2 className="size-3.5 text-ok" />
        {top1 !== null ? <span className="text-ink">Top-1 {fmtPct(top1)}</span> : 'Predicted'}
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1 text-ink-faint">
      <CircleDashed className="size-3.5" /> Not run
    </span>
  )
}

type Filter = 'all' | 'not-run' | 'done' | 'failed'
type Sort = 'id' | 'points-desc' | 'points-asc'
const SORT_LABEL: Record<Sort, string> = { id: 'Trip ID', 'points-desc': 'Most points', 'points-asc': 'Fewest points' }
const SORT_NEXT: Record<Sort, Sort> = { id: 'points-desc', 'points-desc': 'points-asc', 'points-asc': 'id' }

/** Evaluation trips exactly as listed by GET /trips, with client-side search, filter and sort. */
export function TripList({ className }: { className?: string }) {
  const { trips, loadTrips, selectedTripId, selectTrip, predictions, health } = useSession()
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [sort, setSort] = useState<Sort>('id')
  const maxPoints = Math.max(1, ...trips.trips.map((t) => t.n_points))

  const counts = useMemo(() => {
    const c = { all: trips.trips.length, 'not-run': 0, done: 0, failed: 0 }
    for (const t of trips.trips) {
      const s = tripStatus(predictions[t.trip_id])
      if (s === 'done') c.done++
      else if (s === 'failed') c.failed++
      else if (s === 'not-run') c['not-run']++
    }
    return c
  }, [trips.trips, predictions])

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    const list = trips.trips.filter((t) => {
      if (q && !t.trip_id.toLowerCase().includes(q)) return false
      if (filter === 'all') return true
      return tripStatus(predictions[t.trip_id]) === filter
    })
    if (sort === 'points-desc') list.sort((a, b) => b.n_points - a.n_points)
    else if (sort === 'points-asc') list.sort((a, b) => a.n_points - b.n_points)
    else list.sort((a, b) => a.trip_id.localeCompare(b.trip_id))
    return list
  }, [trips.trips, predictions, query, filter, sort])

  const ready = trips.trips.length > 0

  let body
  if (trips.status === 'error') {
    body = (
      <div className="px-4 pb-4">
        <ApiErrorNotice error={trips.error} onRetry={loadTrips} />
      </div>
    )
  } else if ((trips.status === 'loading' || trips.status === 'idle') && !ready) {
    body =
      health.status === 'offline' ? (
        <EmptyState icon={FolderSearch} title="Waiting for the API">
          Trips load automatically once the backend answers. Set the API address from the status button in the top bar.
        </EmptyState>
      ) : (
        <div className="space-y-2 px-3 pb-3" aria-busy="true" aria-label="Loading trips">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-[58px] w-full bg-raised" />
          ))}
        </div>
      )
  } else if (!ready) {
    body = (
      <EmptyState icon={FolderSearch} title="No trips found">
        The API found no trip folders. On Laptop 1, set <code>PATHSNAP_EVAL_DIR</code> to the evaluation folder and restart the API.
      </EmptyState>
    )
  } else if (visible.length === 0) {
    body = (
      <div className="px-4 py-6 text-center text-[13px] text-ink-muted">
        <p>No trips match.</p>
        <button
          type="button"
          className="mt-1.5 text-route underline-offset-4 hover:underline"
          onClick={() => {
            setQuery('')
            setFilter('all')
          }}
        >
          Clear search and filter
        </button>
      </div>
    )
  } else {
    body = (
      <ul className="scroll-thin min-h-0 flex-1 space-y-1 overflow-y-auto px-2 pb-2" aria-label="Evaluation trips">
        {visible.map((t) => {
          const active = t.trip_id === selectedTripId
          return (
            <li key={t.trip_id}>
              <button
                type="button"
                onClick={() => selectTrip(t.trip_id)}
                aria-pressed={active}
                aria-label={`Trip ${t.trip_id}, ${fmtInt(t.n_points)} GPS points`}
                className={cn(
                  'group relative w-full rounded-md border px-3 py-2 text-left transition-colors',
                  active ? 'border-route/55 bg-route/[0.09]' : 'border-transparent hover:border-hairline hover:bg-raised/60',
                )}
              >
                {active && <span className="absolute top-2 bottom-2 left-0 w-[3px] rounded-r-full bg-route" aria-hidden="true" />}
                <div className="flex items-baseline justify-between gap-2">
                  <span className={cn('num text-[18px] leading-none font-semibold', active ? 'text-ink' : 'text-ink/90')}>{t.trip_id}</span>
                  <span className="text-[12px]">
                    <RowStatus rec={predictions[t.trip_id]} />
                  </span>
                </div>
                <div className="mt-1.5 flex items-center gap-2">
                  <div className="h-[4px] flex-1 overflow-hidden rounded-full bg-hairline/70" aria-hidden="true">
                    <div className="h-full rounded-full bg-gps" style={{ width: `${(t.n_points / maxPoints) * 100}%` }} />
                  </div>
                  <span className="w-[84px] shrink-0 text-right text-[12px] text-ink-muted">{fmtInt(t.n_points)} points</span>
                </div>
              </button>
            </li>
          )
        })}
      </ul>
    )
  }

  return (
    <section className={cn('flex min-h-0 flex-col rounded-[10px] border border-hairline bg-surface', className)} aria-label="Trip list">
      <div className="flex items-center justify-between px-4 pt-3.5 pb-2">
        <div>
          <h2 className="font-heading text-[18px] leading-tight font-semibold">Evaluation trips</h2>
          <p className="text-[12.5px] text-ink-muted">{trips.status === 'ready' ? `${trips.trips.length} listed by GET /trips` : 'Loaded from GET /trips'}</p>
        </div>
        <Button
          size="icon-sm"
          variant="ghost"
          onClick={loadTrips}
          disabled={trips.status === 'loading' || health.status !== 'online'}
          aria-label="Reload trip list"
          title="Reload trip list"
        >
          <RotateCw className={trips.status === 'loading' ? 'animate-spin' : ''} />
        </Button>
      </div>

      {ready && (
        <div className="space-y-2 px-3 pb-2.5">
          <div className="relative">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-ink-faint" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search trip ID"
              aria-label="Search trips by ID"
              className="h-8 w-full rounded-md border border-hairline bg-inset pr-7 pl-8 text-[13px] text-ink placeholder:text-ink-faint focus:border-route/60 focus:outline-none [&::-webkit-search-cancel-button]:hidden"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery('')}
                aria-label="Clear search"
                className="absolute top-1/2 right-1.5 grid size-5 -translate-y-1/2 place-items-center rounded text-ink-faint hover:text-ink"
              >
                <X className="size-3.5" />
              </button>
            )}
          </div>
          <div className="flex items-center justify-between gap-1.5">
            <Segmented<Filter>
              label="Filter trips by status"
              size="xs"
              value={filter}
              onChange={setFilter}
              options={[
                { value: 'all', label: 'All', count: counts.all },
                { value: 'not-run', label: 'To do', count: counts['not-run'] },
                { value: 'done', label: 'Done', count: counts.done },
                { value: 'failed', label: 'Failed', count: counts.failed },
              ]}
            />
            <button
              type="button"
              onClick={() => setSort((s) => SORT_NEXT[s])}
              className="flex h-6 shrink-0 items-center gap-1 rounded text-[12px] text-ink-muted hover:text-ink"
              title={`Sorted by ${SORT_LABEL[sort].toLowerCase()}. Click to change.`}
              aria-label={`Sort trips. Currently ${SORT_LABEL[sort]}`}
            >
              <ArrowDownUp className="size-3.5" />
            </button>
          </div>
        </div>
      )}
      {body}
    </section>
  )
}
