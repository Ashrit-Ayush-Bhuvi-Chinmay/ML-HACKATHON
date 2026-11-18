import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router'
import { ArrowDown, ArrowUp, ChartColumn, MapPin } from 'lucide-react'
import { ApiError } from '@/api/client'
import { getExperiments } from '@/api/services'
import type { ExperimentRow, ExperimentsResponse, ModelInfo, PredictOptions } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { InfoTip } from '@/components/common/InfoTip'
import { Notice } from '@/components/common/Notice'
import { Panel } from '@/components/common/Panel'
import { Segmented } from '@/components/common/Segmented'
import { MetricTiles } from '@/components/metrics/MetricTiles'
import { SessionChart, type SessionRow } from '@/components/metrics/SessionChart'
import { TimingChart } from '@/components/metrics/TimingChart'
import { StubNotice } from '@/components/trips/PredictionPanel'
import { TripContextBar } from '@/components/trips/TripContextBar'
import { Button } from '@/components/ui/button'
import { fmtClock, fmtInt, fmtMs, fmtPct } from '@/lib/format'
import { viewOf } from '@/lib/predictionView'
import { toast } from '@/lib/toast'
import { cn } from '@/lib/utils'
import { runKeyOf, useSelectedPrediction, useSession } from '@/state/session'

function OutputFacts({ items }: { items: { label: string; value: string; sub: string; hint: string }[] }) {
  return (
    <dl className="grid grid-cols-2 gap-2 lg:grid-cols-4">
      {items.map((i) => (
        <div key={i.label} className="rounded-lg border border-hairline bg-raised/45 px-3 py-2.5">
          <dt className="flex items-center gap-1 text-[12.5px] text-ink-muted">
            {i.label}
            <InfoTip label={`About ${i.label}`}>{i.hint}</InfoTip>
          </dt>
          <dd className="num mt-1.5 text-[25px] leading-none font-semibold">{i.value}</dd>
          <p className="mt-1 text-[12px] text-ink-faint">{i.sub}</p>
        </div>
      ))}
    </dl>
  )
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const text = (v: unknown): string | null => (typeof v === 'string' && v.trim() ? v : null)

const ABLATION_ORDER = ['classifier_only', 'classifier_hmm', 'distance_hmm', 'distance_only']
const ABLATION_LABELS: Record<string, string> = {
  classifier_only: 'Classifier only',
  classifier_hmm: 'Classifier + HMM',
  distance_hmm: 'Distance + HMM',
  distance_only: 'Distance only',
}

function Leaderboard({
  models,
  status,
  onShowOnMap,
  pending,
  canShow,
}: {
  models: ModelInfo[]
  status: 'idle' | 'loading' | 'ready' | 'error'
  onShowOnMap: (name: string) => void
  pending: string | null
  canShow: boolean
}) {
  const [dir, setDir] = useState<1 | -1>(-1)
  const rows = useMemo(
    () => [...models].sort((a, b) => ((a.test_top1 ?? -1) - (b.test_top1 ?? -1)) * dir),
    [models, dir],
  )

  if (status === 'loading' && !models.length) {
    return <p className="px-4 py-4 text-[13px] text-ink-muted">Loading /models…</p>
  }
  if (status === 'error' && !models.length) {
    return (
      <div className="px-4 py-4">
        <Notice tone="warn" title="Could not load the model list">
          GET /models failed. The picker falls back to the deployed default model.
        </Notice>
      </div>
    )
  }
  if (!rows.length) return <p className="px-4 py-4 text-[13px] text-ink-muted">The API returned no models.</p>

  const th = 'px-3 py-2 text-right font-medium whitespace-nowrap'
  const td = 'px-3 py-2.5 text-right whitespace-nowrap'
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="w-full min-w-[860px] text-[13px]">
        <thead className="bg-inset/60 text-[12px] text-ink-faint">
          <tr className="border-y border-hairline">
            <th className="py-2 pr-3 pl-4 text-left font-medium">Model</th>
            <th className={th}>CV Top-1</th>
            <th className={th} aria-sort={dir === -1 ? 'descending' : 'ascending'}>
              <button
                type="button"
                className="inline-flex flex-row-reverse items-center gap-1 text-ink hover:text-ink-muted"
                onClick={() => setDir((d) => (d === -1 ? 1 : -1))}
              >
                Test Top-1
                {dir === -1 ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />}
              </button>
            </th>
            <th className={th}>Test Seq F1</th>
            <th className={th}>Test Coverage F1</th>
            <th className={th}>Infer / 1k pts</th>
            <th className="py-2 pr-4 pl-3 text-right font-medium">Map</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => {
            const deployed = m.name === 'default'
            const family = text(m.family) ?? text(m.description) ?? text(m.model)
            return (
              <tr key={m.name} className={cn('border-b border-hairline/50 last:border-0', deployed && 'bg-route/[0.06]')}>
                <td className="py-2.5 pr-3 pl-4">
                  <div className="flex items-center gap-1.5">
                    <span className="num text-[14px] font-semibold">{m.name}</span>
                    {deployed && <span className="rounded border border-route/45 px-1 text-[10.5px] text-route">deployed</span>}
                  </div>
                  {family && <p className="text-[11.5px] leading-tight text-ink-faint">{family}</p>}
                </td>
                <td className={cn(td, 'text-ink-muted')}>{fmtPct(num(m.cv_top1))}</td>
                <td className={cn(td, num(m.test_top1) === null ? 'text-ink-faint' : 'font-semibold text-ink')}>{fmtPct(num(m.test_top1))}</td>
                <td className={cn(td, num(m.test_sequence_f1) === null ? 'text-ink-faint' : 'text-ink')}>{fmtPct(num(m.test_sequence_f1))}</td>
                <td className={cn(td, num(m.test_coverage_f1) === null ? 'text-ink-faint' : 'text-ink')}>{fmtPct(num(m.test_coverage_f1))}</td>
                <td className={cn(td, 'text-ink-muted')}>{fmtMs(num(m.infer_ms_per_1k_points))}</td>
                <td className="py-2 pr-4 pl-3 text-right">
                  <Button
                    size="xs"
                    variant="ghost"
                    disabled={!canShow || pending !== null}
                    onClick={() => onShowOnMap(m.name)}
                    title={canShow ? 'Run this model on the selected trip and show it on the map' : 'The API is offline'}
                    aria-label={`Show ${m.name} on the map`}
                  >
                    <MapPin />
                    {pending === m.name ? 'Running…' : 'Show on map'}
                  </Button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Ablation({ tables }: { tables: Record<string, ExperimentRow[]> }) {
  const keys = useMemo(() => Object.keys(tables), [tables])
  const [chosen, setChosen] = useState('')
  const activeModel = keys.includes(chosen) ? chosen : keys.includes('lightgbm_tuned') ? 'lightgbm_tuned' : (keys[0] ?? '')

  const rows = useMemo(() => {
    const list = [...(tables[activeModel] ?? [])]
    const rank = (v: unknown) => {
      const i = ABLATION_ORDER.indexOf(String(v))
      return i === -1 ? ABLATION_ORDER.length : i
    }
    return list.sort((a, b) => rank(a.variant) - rank(b.variant))
  }, [tables, activeModel])

  if (!keys.length) return <p className="px-4 py-4 text-[13px] text-ink-muted">The API returned no HMM ablation tables.</p>

  const th = 'px-3 py-2 text-right font-medium whitespace-nowrap'
  const td = 'px-3 py-2.5 text-right whitespace-nowrap'
  return (
    <div>
      {keys.length > 1 && (
        <div className="flex flex-wrap items-center gap-2 px-4 pb-3">
          <span className="text-[12.5px] text-ink-muted">Model</span>
          <Segmented
            label="Ablation model"
            size="xs"
            value={activeModel}
            onChange={setChosen}
            options={keys.map((k) => ({ value: k, label: k }))}
          />
        </div>
      )}
      <div className="scroll-thin overflow-x-auto">
        <table className="w-full min-w-[720px] text-[13px]">
          <thead className="bg-inset/60 text-[12px] text-ink-faint">
            <tr className="border-y border-hairline">
              <th className="py-2 pr-3 pl-4 text-left font-medium">Variant</th>
              <th className={th}>Top-1</th>
              <th className={th}>Seq F1</th>
              <th className={th}>Coverage F1</th>
              <th className={th}>Continuity</th>
              <th className={th}>Mismatch</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const variant = String(row.variant ?? `run ${i + 1}`)
              return (
                <tr key={`${variant}-${i}`} className="border-b border-hairline/50 last:border-0">
                  <td className="py-2.5 pr-3 pl-4 text-ink">{ABLATION_LABELS[variant] ?? variant}</td>
                  <td className={cn(td, num(row.top1) === null ? 'text-ink-faint' : 'font-semibold text-ink')}>{fmtPct(num(row.top1))}</td>
                  <td className={cn(td, num(row.sequence_f1) === null ? 'text-ink-faint' : 'text-ink')}>{fmtPct(num(row.sequence_f1))}</td>
                  <td className={cn(td, num(row.coverage_f1) === null ? 'text-ink-faint' : 'text-ink')}>{fmtPct(num(row.coverage_f1))}</td>
                  <td className={cn(td, num(row.continuity_pooled) === null ? 'text-ink-faint' : 'text-ink')}>{fmtPct(num(row.continuity_pooled))}</td>
                  <td className={cn(td, num(row.route_mismatch) === null ? 'text-ink-faint' : 'text-ink')}>{fmtPct(num(row.route_mismatch))}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function ModelAnalytics() {
  const {
    predictions,
    trips,
    selectTrip,
    models,
    modelsStatus,
    apiUrl,
    selectedTripId,
    health,
    fetchRun,
    overlayKeysFor,
    setOverlayKeysFor,
    runPrediction,
  } = useSession()
  const rec = useSelectedPrediction()
  const result = rec?.result
  const view = useMemo(() => (result ? viewOf(result.response) : null), [result])
  const [sessionView, setSessionView] = useState<'chart' | 'table'>('chart')
  const [experiments, setExperiments] = useState<ExperimentsResponse | null>(null)
  const [experimentsError, setExperimentsError] = useState<string | null>(null)
  const [pendingModel, setPendingModel] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    const controller = new AbortController()
    getExperiments(controller.signal)
      .then((data) => {
        setExperiments(data)
        setExperimentsError(null)
      })
      .catch((e: unknown) => {
        if (e instanceof ApiError && e.kind === 'cancelled') return
        setExperimentsError(e instanceof Error ? e.message : 'Could not load GET /experiments.')
      })
    return () => controller.abort()
  }, [apiUrl])

  const showModelOnMap = async (name: string) => {
    if (!selectedTripId) {
      toast('info', 'Choose a trip first', 'Select a trip in the trip explorer, then add a model to its map.')
      return
    }
    if (health.status !== 'online') return

    const options: PredictOptions = { model: name === 'default' ? 'default' : name, hmm: true }
    const current = predictions[selectedTripId]?.result
    setPendingModel(name)
    try {
      if (!current) {
        // Nothing on the map yet: this run becomes the current route.
        const outcome = await runPrediction(selectedTripId, options)
        if (outcome !== 'success') throw new Error('The prediction failed; the trip explorer shows the reason.')
      } else {
        const run = await fetchRun(selectedTripId, options)
        if (runKeyOf(run) === runKeyOf(current)) {
          toast('info', 'That run is already the current route', 'Pick another model to overlay it.')
        } else {
          const key = runKeyOf(run)
          const keys = [key, ...overlayKeysFor(selectedTripId).filter((k) => k !== key)]
          setOverlayKeysFor(selectedTripId, keys)
          toast('success', `${name === 'default' ? 'Default model' : name} added to the map`, 'It is drawn behind the current route.')
        }
      }
      navigate('/trips')
    } catch (e) {
      if (!(e instanceof ApiError && e.kind === 'cancelled')) {
        toast('error', 'Could not run this model', e instanceof Error ? e.message : 'Unknown error.')
      }
    } finally {
      setPendingModel(null)
    }
  }

  const rows: (SessionRow & { n: number; scored: number | null })[] = useMemo(
    () =>
      trips.trips
        .map((t) => predictions[t.trip_id]?.result)
        .filter((r): r is NonNullable<typeof r> => !!r)
        .map((r) => ({
          trip_id: r.response.trip_id,
          top1: r.response.metrics.top1_accuracy,
          sequenceF1: r.response.metrics.sequence_f1,
          n: r.response.n_points,
          scored: r.response.metrics.n_points_scored,
        })),
    [trips.trips, predictions],
  )
  const plottable = rows.filter((r) => r.top1 !== null || r.sequenceF1 !== null)

  return (
    <div className="flex flex-col gap-4">
      <TripContextBar />

      {result && view ? (
        <>
          {view.placeholder && <StubNotice />}
          <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <Panel
              title={`Evaluation metrics for trip ${result.response.trip_id}`}
              description={`Computed on the API server against the reference route. Received ${fmtClock(result.receivedAt)}${result.model ? `, model "${result.model}"` : ''}.`}
            >
              <MetricTiles metrics={result.response.metrics} size="lg" />
              {result.response.metrics.note && (
                <Notice tone="warn" title="Metric note" className="mt-3">
                  {result.response.metrics.note}
                </Notice>
              )}
              <p className="mt-3 flex items-center gap-2 text-[12px] text-ink-faint">
                <span className="inline-block h-3 w-5 rounded-[3px] border border-dashed border-hairline" aria-hidden="true" />
                Dashed tiles mean the API returned null. A real zero is shown as 0.
              </p>
            </Panel>
            <Panel title="Processing time" description="Stage timings reported by the API in timings_ms.">
              <TimingChart timings={result.response.timings_ms} roundTripMs={result.roundTripMs} />
            </Panel>
          </div>

          <Panel title="Prediction output" description="Counts that describe what the API returned. They are not accuracy metrics.">
            <OutputFacts
              items={[
                {
                  label: 'Matched points',
                  value: fmtPct(view.stats.total ? view.stats.matched / view.stats.total : null),
                  sub: `${fmtInt(view.stats.matched)} of ${fmtInt(view.stats.total)} GPS points`,
                  hint: 'Share of GPS points with a non-null predicted arc.',
                },
                {
                  label: 'Unmatched stretches',
                  value: fmtInt(view.stats.unmatchedRuns),
                  sub: `longest is ${fmtInt(view.stats.longestUnmatched)} points`,
                  hint: 'Runs of consecutive points with no predicted arc.',
                },
                {
                  label: 'Distinct arcs',
                  value: fmtInt(view.stats.distinctArcs),
                  sub: `${fmtInt(view.stats.matchedRuns)} arc runs`,
                  hint: 'Different arc ids used across predicted_arcs.',
                },
                {
                  label: 'Route arcs',
                  value: fmtInt(result.response.route_arcs.length),
                  sub: 'in the predicted route, split at GPS gaps',
                  hint: 'Number of arcs in the predicted route, split into pieces at GPS gaps.',
                },
              ]}
            />
          </Panel>
        </>
      ) : (
        <Panel>
          <EmptyState
            icon={ChartColumn}
            title="No prediction for the selected trip yet"
            action={
              <Button asChild>
                <Link to="/trips">Open trip explorer</Link>
              </Button>
            }
          >
            Metrics and stage timings appear here after the API returns a prediction. Choose a trip above, or run one in the trip explorer.
          </EmptyState>
        </Panel>
      )}

      <Panel
        title="Trips predicted in this session"
        description="Top-1 accuracy and sequence F1 per trip, exactly as returned by the API."
        actions={
          rows.length > 0 && (
            <Segmented
              label="Session results view"
              size="xs"
              value={sessionView}
              onChange={setSessionView}
              options={[
                { value: 'chart', label: 'Chart' },
                { value: 'table', label: 'Table' },
              ]}
            />
          )
        }
      >
        {rows.length === 0 ? (
          <EmptyState icon={ChartColumn} title="No trips predicted yet" className="py-6">
            Run predictions in the trip explorer, or run every trip from the evaluation page, to compare them here.
          </EmptyState>
        ) : sessionView === 'table' ? (
          <div className="scroll-thin overflow-x-auto">
            <table className="w-full min-w-[520px] text-[13px]">
              <thead className="text-left text-[12px] text-ink-faint">
                <tr className="border-b border-hairline">
                  <th className="py-2 pr-3 font-medium">Trip</th>
                  <th className="px-3 py-2 text-right font-medium">GPS points</th>
                  <th className="px-3 py-2 text-right font-medium">Top-1 accuracy</th>
                  <th className="px-3 py-2 text-right font-medium">Sequence F1</th>
                  <th className="py-2 pl-3 text-right font-medium">Points scored</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.trip_id} className="cursor-pointer border-b border-hairline/50 hover:bg-raised/50" onClick={() => selectTrip(r.trip_id)}>
                    <td className="num py-1.5 pr-3 text-[15px] font-semibold">{r.trip_id}</td>
                    <td className="px-3 py-1.5 text-right text-ink-muted">{fmtInt(r.n)}</td>
                    <td className={r.top1 === null ? 'px-3 py-1.5 text-right text-ink-faint' : 'px-3 py-1.5 text-right'}>{fmtPct(r.top1)}</td>
                    <td className={r.sequenceF1 === null ? 'px-3 py-1.5 text-right text-ink-faint' : 'px-3 py-1.5 text-right'}>{fmtPct(r.sequenceF1)}</td>
                    <td className="py-1.5 pl-3 text-right text-ink-muted">{fmtInt(r.scored)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : plottable.length ? (
          <>
            <SessionChart rows={plottable} />
            {plottable.length < rows.length && (
              <p className="mt-2 text-[12px] text-ink-faint">
                {rows.length - plottable.length} predicted trip(s) returned no metric values and are left out of the chart. The table lists them.
              </p>
            )}
          </>
        ) : (
          <EmptyState icon={ChartColumn} title="No metric values returned yet" className="py-6">
            The API returned N/A for every metric so far, which is expected while the stub pipeline is running. The table view lists the trips.
          </EmptyState>
        )}
      </Panel>

      <Panel
        title="Model leaderboard"
        description="Every model saved on Laptop 1, from GET /models. Sorted by student-test Top-1; the deployed model is highlighted."
        flush
      >
        <Leaderboard
          models={models}
          status={modelsStatus}
          onShowOnMap={(name) => void showModelOnMap(name)}
          pending={pendingModel}
          canShow={health.status === 'online'}
        />
      </Panel>

      <Panel
        title="Classifier versus HMM"
        description="From GET /experiments: classifier-only, classifier + HMM, distance-only and distance + HMM on the student-test trips."
        flush
      >
        {experimentsError ? (
          <div className="px-4 py-4">
            <Notice tone="warn" title="Could not load the ablation tables">
              {experimentsError}
            </Notice>
          </div>
        ) : experiments === null ? (
          <p className="px-4 py-4 text-[13px] text-ink-muted">Loading /experiments…</p>
        ) : (
          <Ablation tables={experiments.hmm_ablation} />
        )}
      </Panel>
    </div>
  )
}
