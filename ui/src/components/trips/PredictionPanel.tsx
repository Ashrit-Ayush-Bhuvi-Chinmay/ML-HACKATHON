import { useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router'
import { ChartColumn, CheckCircle2, CircleDashed, CircleX, FlaskConical, Loader2, Play, Route, Square } from 'lucide-react'
import { ApiErrorNotice } from '@/components/common/ApiErrorNotice'
import { InfoTip } from '@/components/common/InfoTip'
import { Notice } from '@/components/common/Notice'
import { allMetricsNull, MetricTiles } from '@/components/metrics/MetricTiles'
import { StageStrip } from '@/components/metrics/StageStrip'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { useNow } from '@/hooks/useNow'
import { fmtClock, fmtInt, fmtMs, fmtPct, fmtRelative, STAGE_LABELS, STAGE_ORDER } from '@/lib/format'
import { viewOf } from '@/lib/predictionView'
import { cn } from '@/lib/utils'
import { useSession, type PredictionRecord } from '@/state/session'
import { tripStatus } from './TripList'

export function StubNotice({ compact = false }: { compact?: boolean }) {
  return (
    <Notice tone="stub" title="Placeholder output from the stub pipeline">
      {compact
        ? 'No arcs were matched. The dashed line is the raw GPS track, not a reconstructed route.'
        : 'The API returned no matched arcs and no route, so this is not an ML reconstruction. The dashed line on the map is the raw GPS track joined up. Metrics stay N/A until the trained pipeline is connected.'}
    </Notice>
  )
}

export function ResponseWarnings({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null
  return (
    <Notice tone="warn" title={`Response notes (${warnings.length})`}>
      <ul className="list-disc space-y-0.5 pl-4">
        {warnings.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
    </Notice>
  )
}

export function StatusPill({ rec, placeholder }: { rec?: PredictionRecord; placeholder?: boolean }) {
  const s = tripStatus(rec)
  const map = {
    'not-run': { icon: CircleDashed, text: 'Not run', cls: 'border-hairline text-ink-muted' },
    running: { icon: Loader2, text: 'Predicting', cls: 'border-route/50 text-route' },
    done: placeholder
      ? { icon: FlaskConical, text: 'Placeholder', cls: 'border-warn/45 text-warn' }
      : { icon: CheckCircle2, text: 'Predicted', cls: 'border-ok/45 text-ok' },
    failed: { icon: CircleX, text: 'Failed', cls: 'border-err/50 text-err' },
  }[s]
  const Icon = map.icon
  return (
    <span className={cn('inline-flex h-6 items-center gap-1 rounded-full border px-2 text-[12px] font-medium', map.cls)}>
      <Icon className={cn('size-3.5', s === 'running' && 'animate-spin')} /> {map.text}
    </span>
  )
}

/** While waiting: elapsed time only. The API does not stream progress, so stages are not animated. */
function Waiting({ rec, nPoints }: { rec: PredictionRecord; nPoints: number }) {
  const now = useNow(true, 100)
  const elapsed = Math.max(0, now - rec.startedAt) / 1000
  return (
    <div className="rounded-lg border border-route/35 bg-route/[0.06] p-3" aria-live="polite">
      <div className="flex items-center justify-between gap-2 text-[13px]">
        <span className="flex items-center gap-2 text-ink">
          <Loader2 className="size-4 animate-spin text-route" /> Waiting for POST /predict
        </span>
        <span className="num text-[20px] font-semibold text-ink">{elapsed.toFixed(1)} s</span>
      </div>
      <div className="mt-2.5 h-1 overflow-hidden rounded-full bg-hairline">
        <div className="h-full w-1/3 animate-[ps-indeterminate_1.4s_ease-in-out_infinite] rounded-full bg-route" />
      </div>
      <p className="mt-2.5 text-[12.5px] leading-snug text-ink-muted">
        Laptop 1 is processing {fmtInt(nPoints)} GPS points through these stages. Their timings arrive with the response.
      </p>
      <ol className="mt-1.5 flex flex-wrap gap-1">
        {STAGE_ORDER.map((k, i) => (
          <li key={k} className="rounded border border-hairline bg-inset px-1.5 py-0.5 text-[11.5px] text-ink-muted">
            {i + 1} {STAGE_LABELS[k]}
          </li>
        ))}
      </ol>
    </div>
  )
}

function Fact({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <dt className="flex items-center gap-1 text-ink-muted">
        {label}
        {hint && <InfoTip label={`About ${label}`}>{hint}</InfoTip>}
      </dt>
      <dd className="text-right text-ink">{value}</dd>
    </div>
  )
}

/** Inspector for the selected trip: facts, run control, live status, headline results. */
export function PredictionPanel() {
  const { selectedTrip, predictions, runPrediction, cancelPrediction, health, batch, models, modelsStatus } = useSession()
  const [model, setModel] = useState('default')
  const [hmm, setHmm] = useState(true)

  const rec = selectedTrip ? predictions[selectedTrip.trip_id] : undefined
  const result = rec?.result

  // Keep the picker in step with the result on screen, including runs started from the map.
  useEffect(() => {
    if (!result?.response.model) return
    const deployed = health.status === 'online' ? health.data.model : null
    setModel(result.response.model === deployed ? 'default' : result.response.model)
    if (typeof result.response.hmm === 'boolean') setHmm(result.response.hmm)
  }, [result, health])

  if (!selectedTrip) {
    return (
      <div className="px-4 py-6 text-[13px] text-ink-muted">
        <p className="font-heading text-[18px] font-semibold text-ink">Choose a trip</p>
        <p className="mt-1 leading-relaxed">
          Pick one of the evaluation trips on the left, then run the prediction. The API on Laptop 1 matches its GPS track to the road
          network and sends back the route and metrics.
        </p>
      </div>
    )
  }

  const running = rec?.status === 'running'
  const offline = health.status !== 'online'
  const view = result ? viewOf(result.response) : null
  const placeholder = view?.placeholder ?? false
  const modelValue = model === 'default' || models.some((m) => m.name === model) ? model : 'default'

  return (
    <div className="flex flex-col gap-3.5 p-4">
      <div>
        <div className="flex items-center justify-between gap-2">
          <p className="text-[12.5px] text-ink-muted">Selected trip</p>
          <StatusPill rec={rec} placeholder={placeholder} />
        </div>
        <p className="num mt-0.5 text-[36px] leading-none font-semibold">{selectedTrip.trip_id}</p>
      </div>

      <div className="rounded-lg border border-hairline bg-inset px-3 py-2.5 text-[13px]">
        <label className="flex items-center justify-between gap-2">
          <span className="text-ink-muted">Model</span>
          <select
            value={modelValue}
            onChange={(e) => setModel(e.target.value)}
            disabled={running}
            aria-label="Model"
            className="h-7 max-w-[190px] rounded border border-hairline bg-surface px-1.5 text-[12.5px] text-ink"
          >
            <option value="default">default (deployed)</option>
            {models
              .filter((m) => m.name !== 'default')
              .map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
          </select>
        </label>
        <label className="mt-2 flex items-center justify-between gap-2">
          <span className="text-ink-muted">HMM decoding</span>
          <span className="flex items-center gap-2">
            <span className="text-[12px] text-ink-faint">{hmm ? 'on' : 'off'}</span>
            <Switch size="sm" checked={hmm} onCheckedChange={setHmm} disabled={running} aria-label="HMM decoding" />
          </span>
        </label>
        {modelsStatus === 'error' && <p className="mt-1.5 text-[11.5px] text-ink-faint">GET /models failed; only the deployed default is listed.</p>}
      </div>

      <dl className="divide-y divide-hairline/70 rounded-lg border border-hairline bg-inset px-3 py-1 text-[13px]">
        <Fact label="GPS observations" value={fmtInt(selectedTrip.n_points)} hint="From GET /trips." />
        {result && view && (
          <>
            <Fact
              label="Matched points"
              value={
                <>
                  {fmtInt(view.stats.matched)} <span className="text-ink-faint">({fmtPct(view.stats.total ? view.stats.matched / view.stats.total : null)})</span>
                </>
              }
              hint="Points with a non-null predicted arc. A count of the prediction, not an accuracy metric."
            />
            <Fact label="Route arcs" value={fmtInt(result.response.route_arcs.length)} hint="Length of the predicted route, split into pieces at GPS gaps." />
            {typeof result.response.n_route_pieces === 'number' && (
              <Fact label="Route pieces" value={fmtInt(result.response.n_route_pieces)} hint="The route is split into separate lines at GPS gaps." />
            )}
            <Fact label="Model" value={result.response.model ?? result.model ?? '—'} hint="Model name reported with this prediction." />
            <Fact label="Decode" value={result.response.hmm === false ? 'Classifier only' : 'HMM + Viterbi'} hint="Whether HMM/Viterbi decoding produced this route." />
            <Fact label="Server time" value={fmtMs(result.response.timings_ms.total)} hint="timings_ms.total reported by the API." />
            <Fact label="Round trip" value={fmtMs(result.roundTripMs)} hint="Measured in this browser, including network transfer." />
            <Fact label="Received" value={`${fmtClock(result.receivedAt)} (${fmtRelative(result.receivedAt)})`} />
          </>
        )}
      </dl>

      {running ? (
        <Button variant="outline" className="h-10" onClick={() => cancelPrediction(selectedTrip.trip_id)}>
          <Square className="size-3.5" /> Cancel prediction
        </Button>
      ) : (
        <Button
          className="h-10 text-[14px] font-semibold"
          onClick={() => void runPrediction(selectedTrip.trip_id, { model: modelValue, hmm })}
          disabled={offline || batch.running}
          title={offline ? 'The API is offline' : batch.running ? 'A batch run is in progress' : undefined}
        >
          <Play className="size-4" /> {result ? 'Run prediction again' : 'Run prediction'}
        </Button>
      )}
      {offline && !running && <p className="-mt-1.5 text-[12.5px] text-ink-faint">Connect to the API to run predictions.</p>}

      {running && rec && <Waiting rec={rec} nPoints={selectedTrip.n_points} />}
      {rec?.status === 'error' && rec.error && (
        <ApiErrorNotice error={rec.error} onRetry={offline ? undefined : () => void runPrediction(selectedTrip.trip_id, { model: modelValue, hmm })} />
      )}
      {rec?.status === 'error' && result && <p className="text-[12.5px] text-ink-faint">Showing the previous successful result.</p>}

      {result && !running && (
        <>
          {placeholder && <StubNotice compact />}
          <ResponseWarnings warnings={result.warnings} />
          <section aria-label="Processing time">
            <p className="mb-1.5 text-[12.5px] text-ink-muted">Server stages</p>
            <StageStrip timings={result.response.timings_ms} showLegend={false} />
          </section>
          <MetricTiles metrics={result.response.metrics} />
          {result.response.metrics.note && (
            <Notice tone="warn" title="Metric note">
              {result.response.metrics.note}
            </Notice>
          )}
          {allMetricsNull(result.response.metrics) && !placeholder && (
            <p className="text-[12.5px] text-ink-faint">The API returned no metric values for this trip.</p>
          )}
          <div className="grid grid-cols-2 gap-2">
            <Button asChild variant="outline" size="sm">
              <Link to="/route">
                <Route /> Route analysis
              </Link>
            </Button>
            <Button asChild variant="outline" size="sm">
              <Link to="/analytics">
                <ChartColumn /> Analytics
              </Link>
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
