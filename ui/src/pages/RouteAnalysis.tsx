import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router'
import { ChevronLeft, ChevronRight, CircleX, Loader2, MousePointerClick, Play, Route as RouteIcon, Unplug } from 'lucide-react'
import { ApiErrorNotice } from '@/components/common/ApiErrorNotice'
import { EmptyState } from '@/components/common/EmptyState'
import { InfoTip } from '@/components/common/InfoTip'
import { Notice } from '@/components/common/Notice'
import { Panel } from '@/components/common/Panel'
import { Segmented } from '@/components/common/Segmented'
import { RouteMap } from '@/components/map/RouteMap'
import { MatchRibbon } from '@/components/ribbon/MatchRibbon'
import { PointDetails } from '@/components/ribbon/PointDetails'
import { ResponseWarnings, StubNotice } from '@/components/trips/PredictionPanel'
import { TripContextBar } from '@/components/trips/TripContextBar'
import { Button } from '@/components/ui/button'
import { useNow } from '@/hooks/useNow'
import { fmtInt, fmtPct } from '@/lib/format'
import { viewOf } from '@/lib/predictionView'
import type { ArcRun } from '@/lib/sequence'
import { cn } from '@/lib/utils'
import { useSelectedPrediction, useSession } from '@/state/session'

const PAGE = 40
const ROUTE_PREVIEW = 120

function Stat({ label, value, sub, hint }: { label: string; value: string; sub?: string; hint?: string }) {
  return (
    <div className="bg-surface px-4 py-3">
      <dt className="flex items-center gap-1 text-[12.5px] text-ink-muted">
        {label}
        {hint && <InfoTip label={`About ${label}`}>{hint}</InfoTip>}
      </dt>
      <dd className="num mt-1 text-[27px] leading-none font-semibold">{value}</dd>
      {sub && <p className="mt-1 text-[12px] text-ink-faint">{sub}</p>}
    </div>
  )
}

type RunFilter = 'all' | 'matched' | 'unmatched'
type RunSort = 'order' | 'longest'

/**
 * Arc runs. The selected row is always the run containing the shared selected GPS point, so it
 * stays in sync with the ribbon, map and inspector; the table follows it to the right page.
 */
function RunsTable({ runs, selected, onSelect }: { runs: ArcRun[]; selected: ArcRun | null; onSelect: (r: ArcRun) => void }) {
  const [filter, setFilter] = useState<RunFilter>('all')
  const [sort, setSort] = useState<RunSort>('order')
  const scroller = useRef<HTMLDivElement>(null)
  const selectedRow = useRef<HTMLTableRowElement>(null)

  const indexed = useMemo(() => runs.map((r, i) => ({ ...r, n: i + 1 })), [runs])
  const list = useMemo(() => {
    const l = indexed.filter((r) => (filter === 'all' ? true : filter === 'matched' ? r.arc !== null : r.arc === null))
    if (sort === 'longest') l.sort((a, b) => b.length - a.length || a.start - b.start)
    return l
  }, [indexed, filter, sort])
  const matchedCount = runs.filter((r) => r.arc !== null).length

  const pages = Math.max(1, Math.ceil(list.length / PAGE))
  // Page follows the selection; a page chosen by hand holds until the selection, filter or sort changes.
  const viewKey = `${selected?.start ?? 'none'}|${filter}|${sort}`
  const selIdx = selected ? list.findIndex((r) => r.start === selected.start) : -1
  const autoPage = selIdx >= 0 ? Math.floor(selIdx / PAGE) : 0
  const [manual, setManual] = useState<{ page: number; key: string } | null>(null)
  const safePage = Math.min(manual && manual.key === viewKey ? manual.page : autoPage, pages - 1)
  const setPage = (p: number) => setManual({ page: p, key: viewKey })
  const slice = list.slice(safePage * PAGE, safePage * PAGE + PAGE)

  // Keep the selected row visible inside the table's own scroll area (the page itself does not move).
  useEffect(() => {
    const c = scroller.current
    const r = selectedRow.current
    if (!c || !r) return
    const header = 34
    if (r.offsetTop < c.scrollTop + header || r.offsetTop + r.offsetHeight > c.scrollTop + c.clientHeight) {
      c.scrollTop = Math.max(0, r.offsetTop - c.clientHeight / 2)
    }
  }, [selected?.start, safePage])

  return (
    <div className="flex min-h-0 flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pb-2.5">
        <Segmented<RunFilter>
          label="Filter runs"
          size="xs"
          value={filter}
          onChange={setFilter}
          options={[
            { value: 'all', label: 'All', count: runs.length },
            { value: 'matched', label: 'Matched', count: matchedCount },
            { value: 'unmatched', label: 'Gaps', count: runs.length - matchedCount },
          ]}
        />
        <Segmented<RunSort>
          label="Sort runs"
          size="xs"
          value={sort}
          onChange={setSort}
          options={[
            { value: 'order', label: 'In order' },
            { value: 'longest', label: 'Longest' },
          ]}
        />
      </div>
      <div ref={scroller} className="scroll-thin relative max-h-[360px] overflow-y-auto border-t border-hairline">
        <table className="w-full text-[13px]">
          <thead className="sticky top-0 bg-surface text-left text-[12px] text-ink-faint">
            <tr className="border-b border-hairline">
              <th className="py-2 pr-2 pl-4 font-medium">Run</th>
              <th className="px-2 py-2 font-medium">Arc</th>
              <th className="px-2 py-2 text-right font-medium">GPS points</th>
              <th className="py-2 pr-4 pl-2 text-right font-medium">Count</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((r) => {
              const isSel = selected?.start === r.start
              return (
                <tr
                  key={r.start}
                  ref={isSel ? selectedRow : undefined}
                  onClick={() => onSelect(r)}
                  className={cn('cursor-pointer border-b border-hairline/50 transition-colors last:border-0', isSel ? 'bg-route/[0.1]' : 'hover:bg-raised/60')}
                >
                  <td className="py-1.5 pr-2 pl-4 text-ink-faint">{fmtInt(r.n)}</td>
                  <td className="px-2 py-1.5">
                    <button
                      type="button"
                      className={cn('flex items-center gap-1.5 text-left', r.arc === null ? 'text-ink-muted' : 'text-route')}
                      aria-pressed={isSel}
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelect(r)
                      }}
                    >
                      <span
                        className={cn('size-2 rounded-full', r.arc === null ? 'border border-unmatched' : 'bg-gps')}
                        aria-hidden="true"
                      />
                      {r.arc ?? 'unmatched'}
                    </button>
                  </td>
                  <td className="px-2 py-1.5 text-right text-ink-muted">
                    {fmtInt(r.start + 1)}–{fmtInt(r.end + 1)}
                  </td>
                  <td className="py-1.5 pr-4 pl-2 text-right">{fmtInt(r.length)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <div className="flex items-center justify-between border-t border-hairline px-4 py-2 text-[12.5px] text-ink-muted">
          <span>
            {fmtInt(safePage * PAGE + 1)}–{fmtInt(Math.min(list.length, (safePage + 1) * PAGE))} of {fmtInt(list.length)} runs
          </span>
          <div className="flex gap-1">
            <Button size="icon-xs" variant="ghost" disabled={safePage === 0} onClick={() => setPage(safePage - 1)} aria-label="Previous page">
              <ChevronLeft />
            </Button>
            <Button size="icon-xs" variant="ghost" disabled={safePage >= pages - 1} onClick={() => setPage(safePage + 1)} aria-label="Next page">
              <ChevronRight />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function NotReady() {
  const { selectedTrip, predictions, runPrediction, health, batch } = useSession()
  const rec = selectedTrip ? predictions[selectedTrip.trip_id] : undefined
  const now = useNow(rec?.status === 'running', 100)
  const online = health.status === 'online'

  let body
  if (!selectedTrip) {
    body = (
      <EmptyState
        icon={RouteIcon}
        title="No trip selected"
        action={
          <Button asChild>
            <Link to="/trips">Open trip explorer</Link>
          </Button>
        }
      >
        Route analysis shows the arc the API assigned to every GPS point of one trip. Choose a trip above or in the trip explorer.
      </EmptyState>
    )
  } else if (rec?.status === 'running') {
    body = (
      <EmptyState icon={Loader2} title={`Predicting trip ${selectedTrip.trip_id}`}>
        Waiting for the API: <span className="num text-[16px] text-ink">{(Math.max(0, now - rec.startedAt) / 1000).toFixed(1)} s</span>
      </EmptyState>
    )
  } else if (rec?.status === 'error' && rec.error && rec.error.kind !== 'cancelled') {
    body = (
      <div className="mx-auto max-w-[520px] py-8">
        <EmptyState icon={CircleX} title={`Prediction failed for trip ${selectedTrip.trip_id}`} className="py-4" />
        <ApiErrorNotice error={rec.error} onRetry={online ? () => void runPrediction(selectedTrip.trip_id) : undefined} />
      </div>
    )
  } else if (!online) {
    body = (
      <EmptyState icon={Unplug} title="The API is offline">
        Trip {selectedTrip.trip_id} has no prediction yet. Connect to Laptop 1&rsquo;s API from the status button in the top bar, then run it.
      </EmptyState>
    )
  } else {
    body = (
      <EmptyState
        icon={RouteIcon}
        title={`No prediction for trip ${selectedTrip.trip_id} yet`}
        action={
          <Button onClick={() => void runPrediction(selectedTrip.trip_id)} disabled={batch.running}>
            <Play className="size-4" /> Run prediction
          </Button>
        }
      >
        Run the prediction to see which arc each of its {fmtInt(selectedTrip.n_points)} GPS points was matched to.
      </EmptyState>
    )
  }
  return <Panel className="flex-1">{body}</Panel>
}

export function RouteAnalysis() {
  const { selectedTrip, focusIndex, setFocusIndex } = useSession()
  const rec = useSelectedPrediction()
  const result = rec?.result
  const [showAllRoute, setShowAllRoute] = useState(false)
  const view = useMemo(() => (result ? viewOf(result.response) : null), [result])

  if (!selectedTrip || !result || !view) {
    return (
      <div className="flex flex-1 flex-col gap-4">
        <TripContextBar />
        <NotReady />
      </div>
    )
  }

  const { response } = result
  const { stats, runs, placeholder } = view
  const total = view.pointInfo(0).total
  const focusInfo = focusIndex !== null && focusIndex < total ? view.pointInfo(focusIndex) : null
  const focusArc = focusInfo?.arc ?? null
  // Single source of truth: the selected run is the run containing the selected GPS point.
  const selectedRun = focusInfo?.run ?? null

  const selectRun = (r: ArcRun) => setFocusIndex(selectedRun?.start === r.start ? null : r.start)

  /** First GPS point matched to an arc, used when a route arc is clicked. */
  const firstPointOf = (arc: number) => response.predicted_arcs.indexOf(arc)

  return (
    <div className="flex flex-col gap-4">
      <TripContextBar />
      {placeholder && <StubNotice />}
      <ResponseWarnings warnings={result.warnings} />

      <section aria-label="Sequence summary">
        <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-[10px] border border-hairline bg-hairline sm:grid-cols-3 xl:grid-cols-6">
          <Stat
            label="Matched points"
            value={fmtInt(stats.matched)}
            sub={`${fmtPct(stats.total ? stats.matched / stats.total : null)} of ${fmtInt(stats.total)}`}
            hint="GPS points with a non-null predicted arc."
          />
          <Stat label="Unmatched points" value={fmtInt(stats.unmatched)} sub={`in ${fmtInt(stats.unmatchedRuns)} stretches`} hint="GPS points whose predicted arc is null." />
          <Stat label="Longest gap" value={fmtInt(stats.longestUnmatched)} sub="consecutive unmatched points" />
          <Stat label="Distinct arcs" value={fmtInt(stats.distinctArcs)} sub="used in predicted_arcs" />
          <Stat label="Arc runs" value={fmtInt(stats.matchedRuns)} sub="stretches on one arc" hint="Consecutive GPS points matched to the same arc count as one run." />
          <Stat
            label="Route arcs"
            value={fmtInt(response.route_arcs.length)}
            sub={`continuity ${fmtPct(response.metrics.route_continuity)} from the API`}
            hint="Length of route_arcs, the connected route the API reconstructed."
          />
        </dl>
        <p className="mt-1.5 text-[12px] text-ink-faint">
          These counts describe the prediction itself. Accuracy against the reference route is computed only on the API server.
        </p>
      </section>

      <Panel title="Point-by-point matching" description={`The arc predicted for each of the ${fmtInt(response.n_points)} GPS observations of trip ${response.trip_id}, in time order.`}>
        <MatchRibbon arcs={response.predicted_arcs} height={40} lens />
      </Panel>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_390px]">
        <RouteMap
          result={result}
          highlight={selectedRun ? { start: selectedRun.start, end: selectedRun.end } : null}
          className="h-[56vh] min-h-[420px]"
          compactLegend
          showPointCard={false}
        />
        <div className="flex min-w-0 flex-col gap-4">
          <Panel title="Point inspector" description="What the response says about the selected GPS point.">
            {focusInfo ? (
              <PointDetails
                info={focusInfo}
                onStep={(d) => setFocusIndex(Math.min(Math.max(0, focusInfo.index + d), total - 1))}
                onClose={() => setFocusIndex(null)}
              />
            ) : (
              <div className="flex items-start gap-2.5 rounded-lg border border-dashed border-hairline px-3 py-3 text-[13px] text-ink-muted">
                <MousePointerClick className="mt-0.5 size-4 shrink-0 text-ink-faint" />
                Select a point on the ribbon, the detail strip or the map. Arrow keys on the ribbon step through points.
              </div>
            )}
          </Panel>
          <Panel title="Arc runs" description="Select a run to ring its GPS points on the map." flush bodyClassName="flex flex-col">
            {runs.length ? <RunsTable runs={runs} selected={selectedRun} onSelect={selectRun} /> : <p className="px-4 pb-4 text-[13px] text-ink-muted">No runs.</p>}
          </Panel>
        </div>
      </div>

      <Panel
        title="Reconstructed route"
        description="route_arcs: the connected predicted route as arc ids in travel order. Select an arc to jump to its first matched GPS point."
        actions={
          focusArc !== null && view.routePos.get(focusArc) ? (
            <span className="flex items-center gap-1.5 text-[12.5px] text-ink-muted">
              <span className="size-2.5 rounded-sm bg-route" aria-hidden="true" /> arc {focusArc} of the selected point
            </span>
          ) : undefined
        }
      >
        {response.route_arcs.length ? (
          <>
            <ol className="flex flex-wrap gap-1.5">
              {(showAllRoute ? response.route_arcs : response.route_arcs.slice(0, ROUTE_PREVIEW)).map((arc, i) => {
                const isFocusArc = focusArc === arc
                const first = firstPointOf(arc)
                return (
                  <li key={`${i}-${arc}`}>
                    <button
                      type="button"
                      onClick={() => first >= 0 && setFocusIndex(first)}
                      disabled={first < 0}
                      title={first < 0 ? 'No GPS point was matched to this arc' : `Jump to GPS point ${first + 1}`}
                      className={cn(
                        'flex items-center overflow-hidden rounded-md border text-[12.5px] transition-colors',
                        isFocusArc ? 'border-route bg-route/15' : 'border-hairline hover:border-ink-faint',
                        first < 0 && 'cursor-default opacity-70',
                      )}
                    >
                      <span className="bg-raised px-1.5 py-0.5 text-ink-faint">{i + 1}</span>
                      <span className={cn('px-2 py-0.5', isFocusArc ? 'text-ink' : 'text-route')}>{arc}</span>
                    </button>
                  </li>
                )
              })}
            </ol>
            {response.route_arcs.length > ROUTE_PREVIEW && (
              <Button variant="ghost" size="sm" className="mt-2" onClick={() => setShowAllRoute((v) => !v)}>
                {showAllRoute ? 'Show fewer' : `Show all ${fmtInt(response.route_arcs.length)} arcs`}
              </Button>
            )}
          </>
        ) : (
          <p className="text-[13px] text-ink-muted">The API returned an empty route for this trip.</p>
        )}
      </Panel>

      <Notice tone="info" title="Classifier-only versus HMM comparison">
        API v0.1 returns only the final decoded output, so this page cannot compare classifier-only predictions with HMM/Viterbi decoding.
        That comparison belongs in the technical report (Section 5), produced from the model team&rsquo;s experiment results. Per-point
        timestamps and candidate scores would also need new API fields.
      </Notice>
    </div>
  )
}
