import { useMemo, useState } from 'react'
import { Link } from 'react-router'
import { ChartColumn, FileBarChart2, FlaskConical, GitCompareArrows, SlidersHorizontal } from 'lucide-react'
import { EmptyState } from '@/components/common/EmptyState'
import { InfoTip } from '@/components/common/InfoTip'
import { Panel } from '@/components/common/Panel'
import { Segmented } from '@/components/common/Segmented'
import { MetricTiles } from '@/components/metrics/MetricTiles'
import { SessionChart, type SessionRow } from '@/components/metrics/SessionChart'
import { TimingChart } from '@/components/metrics/TimingChart'
import { StubNotice } from '@/components/trips/PredictionPanel'
import { TripContextBar } from '@/components/trips/TripContextBar'
import { Button } from '@/components/ui/button'
import { fmtClock, fmtInt, fmtPct } from '@/lib/format'
import { viewOf } from '@/lib/predictionView'
import { useSelectedPrediction, useSession } from '@/state/session'

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

const NOT_SERVED = [
  {
    icon: GitCompareArrows,
    title: 'Five-model comparison',
    text: 'Cross-validation and student-test Macro F1, Top-1 accuracy, log loss and inference time for every model tried.',
    where: 'results/ and report Section 3',
  },
  {
    icon: SlidersHorizontal,
    title: 'Bayesian optimisation',
    text: 'Search space, trials, best hyperparameters and tuned versus untuned scores.',
    where: 'results/ and report Section 4',
  },
  {
    icon: FlaskConical,
    title: 'Classifier versus HMM',
    text: 'Classifier-only predictions compared with classifier plus Viterbi decoding.',
    where: 'Report Section 5',
  },
]

export function ModelAnalytics() {
  const { predictions, trips, selectTrip } = useSession()
  const rec = useSelectedPrediction()
  const result = rec?.result
  const view = useMemo(() => (result ? viewOf(result.response) : null), [result])
  const [sessionView, setSessionView] = useState<'chart' | 'table'>('chart')

  const rows: (SessionRow & { n: number; scored: number | null })[] = useMemo(
    () =>
      trips.trips
        .map((t) => predictions[t.trip_id]?.result)
        .filter((r): r is NonNullable<typeof r> => !!r)
        .map((r) => ({
          trip_id: r.response.trip_id,
          top1: r.response.metrics.top1_accuracy,
          f1: r.response.metrics.route_f1,
          n: r.response.n_points,
          scored: r.response.metrics.n_points_scored,
        })),
    [trips.trips, predictions],
  )
  const plottable = rows.filter((r) => r.top1 !== null || r.f1 !== null)

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
                  sub: 'in route_arcs, travel order',
                  hint: 'Length of the connected route the API reconstructed.',
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
        description="Top-1 accuracy and route F1 per trip, exactly as returned by the API."
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
                  <th className="px-3 py-2 text-right font-medium">Route F1</th>
                  <th className="py-2 pl-3 text-right font-medium">Points scored</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.trip_id} className="cursor-pointer border-b border-hairline/50 hover:bg-raised/50" onClick={() => selectTrip(r.trip_id)}>
                    <td className="num py-1.5 pr-3 text-[15px] font-semibold">{r.trip_id}</td>
                    <td className="px-3 py-1.5 text-right text-ink-muted">{fmtInt(r.n)}</td>
                    <td className={r.top1 === null ? 'px-3 py-1.5 text-right text-ink-faint' : 'px-3 py-1.5 text-right'}>{fmtPct(r.top1)}</td>
                    <td className={r.f1 === null ? 'px-3 py-1.5 text-right text-ink-faint' : 'px-3 py-1.5 text-right'}>{fmtPct(r.f1)}</td>
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

      <Panel title="Not served by the API" description="API v0.1 exposes only the deployed model's predictions, so this interface does not show training experiments.">
        <ul className="grid grid-cols-1 gap-2 md:grid-cols-3">
          {NOT_SERVED.map(({ icon: Icon, title, text, where }) => (
            <li key={title} className="rounded-lg border border-dashed border-hairline px-3.5 py-3">
              <p className="flex items-center gap-2 text-[14px] font-medium text-ink">
                <Icon className="size-4 text-ink-faint" /> {title}
              </p>
              <p className="mt-1 text-[12.5px] leading-snug text-ink-muted">{text}</p>
              <p className="mt-2 flex items-center gap-1.5 text-[12px] text-ink-faint">
                <FileBarChart2 className="size-3.5" /> {where}
              </p>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-[12px] text-ink-faint">
          Showing these here would need a new read-only API endpoint serving the model team&rsquo;s recorded results. That is a backend
          enhancement and is not part of API v0.1.
        </p>
      </Panel>
    </div>
  )
}
