import { Fragment, useMemo, useState } from 'react'
import { useNavigate } from 'react-router'
import { ArrowDown, ArrowUp, Check, CircleDashed, CircleX, FlaskConical, ListChecks, Loader2, Map as MapIcon, Play, RotateCw, Square } from 'lucide-react'
import type { PredictionMetrics } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { Panel } from '@/components/common/Panel'
import { Segmented } from '@/components/common/Segmented'
import { RequestConsole } from '@/components/evaluation/RequestConsole'
import { tripStatus, type TripStatus } from '@/components/trips/TripList'
import { Button } from '@/components/ui/button'
import { useNow } from '@/hooks/useNow'
import { fmtInt, fmtMs, fmtPct } from '@/lib/format'
import { viewOf } from '@/lib/predictionView'
import { cn } from '@/lib/utils'
import { useSession, type PredictionRecord } from '@/state/session'

type StepState = 'done' | 'current' | 'pending' | 'blocked'

function Steps() {
  const { health, trips, selectedTrip, predictions } = useSession()
  const rec = selectedTrip ? predictions[selectedTrip.trip_id] : undefined
  const online = health.status === 'online'
  const tripsReady = trips.status === 'ready' && trips.trips.length > 0
  const predicted = !!rec?.result

  const steps: { title: string; detail: string; state: StepState }[] = [
    {
      title: 'Connect to the API',
      detail: online ? `Online, model "${health.data.model}"` : health.status === 'offline' ? 'Unreachable: check the address' : 'Checking',
      state: online ? 'done' : health.status === 'offline' ? 'blocked' : 'current',
    },
    {
      title: 'Load evaluation trips',
      detail: tripsReady ? `${trips.trips.length} trips from GET /trips` : trips.status === 'error' ? 'Trip list failed to load' : 'Waiting for the API',
      state: tripsReady ? 'done' : trips.status === 'error' ? 'blocked' : online ? 'current' : 'pending',
    },
    {
      title: 'Select a trip',
      detail: selectedTrip ? `Trip ${selectedTrip.trip_id}` : 'Choose one in the trip explorer',
      state: selectedTrip ? 'done' : tripsReady ? 'current' : 'pending',
    },
    {
      title: 'Run the prediction',
      detail:
        rec?.status === 'running' ? 'Waiting for POST /predict' : predicted ? 'Prediction received' : rec?.status === 'error' ? 'Last run failed' : 'Not run yet',
      state: rec?.status === 'error' && !predicted ? 'blocked' : predicted ? 'done' : selectedTrip ? 'current' : 'pending',
    },
    {
      title: 'Inspect the results',
      detail: predicted ? 'Map, route analysis and metrics ready' : 'Available after the prediction',
      state: predicted ? 'current' : 'pending',
    },
  ]
  const doneCount = steps.filter((s) => s.state === 'done').length

  return (
    <section aria-label="Evaluation steps" className="rounded-[10px] border border-hairline bg-surface px-4 py-4">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2 className="font-heading text-[18px] font-semibold">Evaluator walkthrough</h2>
        <span className="text-[12.5px] text-ink-muted">
          {doneCount} of {steps.length} steps done
        </span>
      </div>
      <ol className="grid gap-3 sm:grid-cols-5 sm:gap-0">
        {steps.map((s, i) => (
          <li key={s.title} className="relative flex gap-3 sm:flex-col sm:gap-2 sm:pr-4">
            {i < steps.length - 1 && (
              <span
                className={cn('absolute top-3 left-8 hidden h-[2px] w-[calc(100%-2.5rem)] rounded-full sm:block', s.state === 'done' ? 'bg-ok/60' : 'bg-hairline')}
                aria-hidden="true"
              />
            )}
            <span
              className={cn(
                'relative grid size-6 shrink-0 place-items-center rounded-full border text-[12px] font-semibold',
                s.state === 'done' && 'border-ok/60 bg-ok/15 text-ok',
                s.state === 'current' && 'border-route bg-route/15 text-route',
                s.state === 'pending' && 'border-hairline bg-surface text-ink-faint',
                s.state === 'blocked' && 'border-err/60 bg-err/10 text-err',
              )}
              aria-hidden="true"
            >
              {s.state === 'done' ? <Check className="size-3.5" /> : s.state === 'blocked' ? <CircleX className="size-3.5" /> : i + 1}
            </span>
            <div className="min-w-0">
              <p className={cn('text-[13.5px] font-medium', s.state === 'pending' ? 'text-ink-muted' : 'text-ink')}>{s.title}</p>
              <p className="text-[12px] leading-snug text-ink-faint">
                <span className="sr-only">{s.state}: </span>
                {s.detail}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

/** Run controls and live progress. One cell per trip in the queue, coloured by outcome with a legend. */
function BatchRunner() {
  const { trips, predictions, batch, runBatch, cancelBatch, health } = useSession()
  const now = useNow(batch.running, 250)
  const online = health.status === 'online' && trips.status === 'ready'
  const ids = trips.trips.map((t) => t.trip_id)
  const notRun = ids.filter((id) => tripStatus(predictions[id]) === 'not-run')
  const failed = ids.filter((id) => tripStatus(predictions[id]) === 'failed')

  const elapsed = batch.startedAt ? ((batch.finishedAt ?? now) - batch.startedAt) / 1000 : 0
  const remaining = batch.queue.length - batch.done

  return (
    <section aria-label="Run trips" className="rounded-[10px] border border-hairline bg-surface px-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-heading text-[18px] font-semibold">Run evaluation trips</h2>
          <p className="mt-0.5 text-[13px] text-ink-muted">Trips run one after another so each request shows up clearly in the API console on Laptop 1.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          {batch.running ? (
            <Button variant="outline" onClick={cancelBatch}>
              <Square className="size-3" /> Stop after current trip
            </Button>
          ) : (
            <>
              <Button onClick={() => void runBatch(ids, 'all trips')} disabled={!online || !ids.length} className="font-semibold">
                <Play className="size-3.5" /> Run all {ids.length || ''} trips
              </Button>
              {notRun.length > 0 && notRun.length < ids.length && (
                <Button variant="outline" onClick={() => void runBatch(notRun, 'trips not run yet')} disabled={!online}>
                  Run {notRun.length} not run yet
                </Button>
              )}
              {failed.length > 0 && (
                <Button variant="outline" onClick={() => void runBatch(failed, 'failed trips')} disabled={!online}>
                  <RotateCw className="size-3.5" /> Retry {failed.length} failed
                </Button>
              )}
            </>
          )}
        </div>
      </div>

      {batch.queue.length > 0 && (
        <div className="mt-4" aria-live="polite">
          <div className="flex gap-[3px]" role="img" aria-label={`${batch.done} of ${batch.queue.length} trips processed`}>
            {batch.queue.map((id) => {
              const state = batch.succeeded.includes(id) ? 'ok' : batch.failed.includes(id) ? 'fail' : batch.current === id ? 'run' : 'wait'
              return (
                <span
                  key={id}
                  title={`Trip ${id}`}
                  className={cn(
                    'h-2.5 flex-1 rounded-[2px]',
                    state === 'ok' && 'bg-ok',
                    state === 'fail' && 'bg-err',
                    state === 'run' && 'animate-pulse bg-route',
                    state === 'wait' && 'bg-hairline',
                  )}
                />
              )
            })}
          </div>
          <div className="mt-2.5 flex flex-wrap items-center gap-x-5 gap-y-1 text-[13px]">
            <span className="text-ink-muted">
              {batch.running ? (
                <>
                  Running {batch.label}
                  {batch.current && (
                    <>
                      , now <span className="num text-[15px] font-semibold text-ink">{batch.current}</span>
                    </>
                  )}
                </>
              ) : (
                <>
                  Last run: {batch.label}
                  {batch.stopped ? ', stopped early' : ''}
                </>
              )}
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-[2px] bg-ok" aria-hidden="true" /> Succeeded <span className="num text-[15px] font-semibold">{batch.succeeded.length}</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-[2px] bg-err" aria-hidden="true" /> Failed <span className="num text-[15px] font-semibold">{batch.failed.length}</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-[2px] bg-hairline" aria-hidden="true" /> Remaining <span className="num text-[15px] font-semibold">{remaining}</span>
            </span>
            <span className="text-ink-muted">
              Elapsed <span className="num text-[15px] font-semibold text-ink">{elapsed.toFixed(1)} s</span>
            </span>
          </div>
        </div>
      )}
    </section>
  )
}

const COLS: { key: keyof PredictionMetrics; label: string }[] = [
  { key: 'top1_accuracy', label: 'Top-1' },
  { key: 'route_precision', label: 'Precision' },
  { key: 'route_recall', label: 'Recall' },
  { key: 'route_f1', label: 'F1' },
  { key: 'route_continuity', label: 'Continuity' },
]

type SortKey = 'trip' | 'points' | 'status' | keyof PredictionMetrics | 'server'
type Filter = 'all' | TripStatus

const STATUS_ORDER: Record<TripStatus, number> = { failed: 0, running: 1, 'not-run': 2, done: 3 }

function mean(values: (number | null)[]): number | null {
  const v = values.filter((x): x is number => x !== null)
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}

function StatusCell({ rec }: { rec?: PredictionRecord }) {
  const s = tripStatus(rec)
  if (s === 'running')
    return (
      <span className="flex items-center gap-1.5 text-route">
        <Loader2 className="size-3.5 animate-spin" /> Predicting
      </span>
    )
  if (s === 'failed')
    return (
      <span className="flex items-center gap-1.5 text-err">
        <CircleX className="size-3.5" /> Failed
      </span>
    )
  if (s === 'done') {
    const placeholder = viewOf(rec!.result!.response).placeholder
    return placeholder ? (
      <span className="flex items-center gap-1.5 text-warn">
        <FlaskConical className="size-3.5" /> Placeholder
      </span>
    ) : (
      <span className="flex items-center gap-1.5 text-ink">
        <Check className="size-3.5 text-ok" /> Predicted
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1.5 text-ink-faint">
      <CircleDashed className="size-3.5" /> Not run
    </span>
  )
}

function ResultsTable() {
  const { trips, predictions, selectTrip, runPrediction, health, batch } = useSession()
  const navigate = useNavigate()
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'trip', dir: 1 })
  const [filter, setFilter] = useState<Filter>('all')
  const canRun = health.status === 'online' && !batch.running

  const counts = useMemo(() => {
    const c: Record<Filter, number> = { all: trips.trips.length, done: 0, failed: 0, 'not-run': 0, running: 0 }
    for (const t of trips.trips) c[tripStatus(predictions[t.trip_id])]++
    return c
  }, [trips.trips, predictions])

  const rows = useMemo(() => {
    const value = (id: string, n: number, key: SortKey): number | string | null => {
      const rec = predictions[id]
      const r = rec?.result?.response
      if (key === 'trip') return id
      if (key === 'points') return n
      if (key === 'status') return STATUS_ORDER[tripStatus(rec)]
      if (key === 'server') return r?.timings_ms.total ?? null
      return r ? r.metrics[key] : null
    }
    const list = trips.trips.filter((t) => filter === 'all' || tripStatus(predictions[t.trip_id]) === filter)
    return [...list].sort((a, b) => {
      const va = value(a.trip_id, a.n_points, sort.key)
      const vb = value(b.trip_id, b.n_points, sort.key)
      // Missing values always sink to the bottom, whatever the direction.
      if (va === null && vb === null) return 0
      if (va === null) return 1
      if (vb === null) return -1
      return (typeof va === 'string' ? va.localeCompare(vb as string) : va - (vb as number)) * sort.dir
    })
  }, [trips.trips, predictions, sort, filter])

  const done = trips.trips.map((t) => predictions[t.trip_id]?.result).filter((r): r is NonNullable<typeof r> => !!r)

  const head = (k: SortKey, children: string, align: 'left' | 'right' = 'right') => {
    const active = sort.key === k
    return (
      <th key={k} className={cn('px-3 py-2 font-medium whitespace-nowrap', align === 'right' ? 'text-right' : 'text-left')} aria-sort={active ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none'}>
        <button
          type="button"
          onClick={() => setSort((s) => ({ key: k, dir: s.key === k ? ((s.dir * -1) as 1 | -1) : k === 'trip' || k === 'status' ? 1 : -1 }))}
          className={cn('inline-flex items-center gap-1 hover:text-ink', active ? 'text-ink' : 'text-ink-faint', align === 'right' && 'flex-row-reverse')}
        >
          {children}
          {active ? sort.dir === 1 ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" /> : <span className="size-3" />}
        </button>
      </th>
    )
  }

  const td = 'px-3 py-2.5 text-right whitespace-nowrap'

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pb-3">
        <Segmented<Filter>
          label="Filter results"
          size="xs"
          value={filter}
          onChange={setFilter}
          options={[
            { value: 'all', label: 'All', count: counts.all },
            { value: 'done', label: 'Predicted', count: counts.done },
            { value: 'failed', label: 'Failed', count: counts.failed },
            { value: 'not-run', label: 'Not run', count: counts['not-run'] },
          ]}
        />
        <p className="text-[12px] text-ink-faint">Select a column heading to sort. Missing values stay at the bottom.</p>
      </div>
      <div className="scroll-thin overflow-x-auto border-t border-hairline">
        <table className="w-full min-w-[980px] text-[13px]">
          <thead className="bg-inset/60 text-[12px]">
            <tr className="border-b border-hairline">
              {head('trip', 'Trip', 'left')}
              {head('status', 'Status', 'left')}
              {head('points', 'GPS points')}
              {COLS.map((c) => head(c.key, c.label))}
              {head('n_points_scored', 'Scored')}
              {head('server', 'Server time')}
              <th className="py-2 pr-4 pl-3 text-right font-medium text-ink-faint">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((t) => {
              const rec = predictions[t.trip_id]
              const r = rec?.result?.response
              const s = tripStatus(rec)
              return (
                <Fragment key={t.trip_id}>
                  <tr className={cn('border-b border-hairline/50 transition-colors hover:bg-raised/40', s === 'failed' && 'bg-err/[0.05]')}>
                    <td className="py-2.5 pr-3 pl-4">
                      <span className="num text-[16px] font-semibold">{t.trip_id}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <StatusCell rec={rec} />
                    </td>
                    <td className={td}>{fmtInt(t.n_points)}</td>
                    {COLS.map((c) => (
                      <td key={c.key} className={cn(td, r && r.metrics[c.key] === null ? 'text-ink-faint' : 'text-ink')}>
                        {r ? fmtPct(r.metrics[c.key]) : <span className="text-ink-faint/60">–</span>}
                      </td>
                    ))}
                    <td className={td}>{r ? fmtInt(r.metrics.n_points_scored) : <span className="text-ink-faint/60">–</span>}</td>
                    <td className={cn(td, 'text-ink-muted')}>{r ? fmtMs(r.timings_ms.total) : <span className="text-ink-faint/60">–</span>}</td>
                    <td className="py-2 pr-4 pl-3">
                      <div className="flex justify-end gap-1">
                        {(s === 'failed' || s === 'not-run') && (
                          <Button size="xs" variant={s === 'failed' ? 'outline' : 'ghost'} disabled={!canRun} onClick={() => void runPrediction(t.trip_id)}>
                            {s === 'failed' ? <RotateCw /> : <Play />} {s === 'failed' ? 'Retry' : 'Run'}
                          </Button>
                        )}
                        <Button
                          size="xs"
                          variant="ghost"
                          onClick={() => {
                            selectTrip(t.trip_id)
                            navigate('/trips')
                          }}
                          aria-label={`Open trip ${t.trip_id} on the map`}
                        >
                          <MapIcon /> Open
                        </Button>
                      </div>
                    </td>
                  </tr>
                  {s === 'failed' && rec?.error && (
                    <tr className="border-b border-hairline/50 bg-err/[0.05]">
                      <td colSpan={COLS.length + 6} className="px-4 pt-0 pb-2.5 text-[12.5px] text-err">
                        {rec.error.status ? `HTTP ${rec.error.status}: ` : ''}
                        {rec.error.message}
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
          {done.length > 1 && filter === 'all' && (
            <tfoot>
              <tr className="border-t border-hairline bg-raised/35 text-ink">
                <td className="py-2.5 pr-3 pl-4 text-[12.5px] text-ink-muted" colSpan={3}>
                  Mean over {done.length} predicted trips (unweighted, of API values)
                </td>
                {COLS.map((c) => (
                  <td key={c.key} className={cn(td, 'font-medium')}>
                    {fmtPct(mean(done.map((d) => d.response.metrics[c.key])))}
                  </td>
                ))}
                <td colSpan={3} />
              </tr>
            </tfoot>
          )}
        </table>
      </div>
      {rows.length === 0 && <p className="px-4 py-5 text-center text-[13px] text-ink-muted">No trips match this filter.</p>}
    </>
  )
}

export function Evaluation() {
  const { trips } = useSession()
  return (
    <div className="flex flex-col gap-4">
      <Steps />
      <BatchRunner />
      <Panel title="Results by trip" description="Metrics exactly as returned by the API for each trip. N/A means the API returned null; a dash means the trip has not been run." flush>
        {trips.trips.length ? (
          <ResultsTable />
        ) : (
          <EmptyState icon={ListChecks} title="No trips loaded">
            The trip list loads from the API once it is online.
          </EmptyState>
        )}
      </Panel>
      <RequestConsole />
    </div>
  )
}
