import type { ReactNode } from 'react'
import { Link } from 'react-router'
import { ArrowRight, ArrowUpRight, Cpu, FlaskConical, ListChecks, Map as MapIcon, MonitorSmartphone, Server, ShieldCheck, Unplug } from 'lucide-react'
import { Panel } from '@/components/common/Panel'
import { StatusDot } from '@/components/common/StatusDot'
import { StageStrip } from '@/components/metrics/StageStrip'
import { SnapSchematic } from '@/components/overview/SnapSchematic'
import { Button } from '@/components/ui/button'
import { fmtClock, fmtInt, fmtMs, STAGE_LABELS, STAGE_ORDER } from '@/lib/format'
import { cn } from '@/lib/utils'
import { useSession, type PredictionResult } from '@/state/session'

const STAGE_TEXT: Record<(typeof STAGE_ORDER)[number], string> = {
  load: 'Reads the trip’s .track, .nodes and .arcs files.',
  candidates: 'Finds road arcs each GPS point could belong to.',
  features: 'Builds the classifier’s input for each point–arc pair.',
  classifier: 'The selected ML model scores every candidate.',
  hmm: 'Viterbi decoding picks the most likely connected route.',
}

function hostOf(url: string) {
  try {
    return new URL(url).host
  } catch {
    return url
  }
}

/** Three instrument cells: API link, model, trip list. Values come from /health and /trips only. */
function StatusStrip() {
  const { health, trips, apiUrl, isStubModel } = useSession()
  const online = health.status === 'online'

  const cells: { label: string; icon: typeof Server; tone: 'ok' | 'warn' | 'err' | 'idle'; value: ReactNode; sub: ReactNode }[] = [
    {
      label: 'API on Laptop 1',
      icon: online ? Server : Unplug,
      tone: online ? 'ok' : health.status === 'offline' ? 'err' : 'idle',
      value: online ? 'Online' : health.status === 'offline' ? 'Offline' : 'Connecting',
      sub: (
        <>
          {hostOf(apiUrl)}
          {online && <span className="text-ink-faint">, {fmtMs(health.latencyMs)}</span>}
        </>
      ),
    },
    {
      label: 'Model',
      icon: isStubModel ? FlaskConical : Cpu,
      tone: !health.data ? 'idle' : isStubModel ? 'warn' : 'ok',
      value: health.data ? (isStubModel ? 'Stub' : health.data.model) : 'Unknown',
      sub: health.data ? (isStubModel ? 'Placeholder, not trained' : 'Reported by /health') : 'Known once the API answers',
    },
    {
      label: 'Evaluation trips',
      icon: ListChecks,
      tone: trips.status === 'ready' && trips.trips.length ? 'ok' : 'idle',
      value: trips.status === 'ready' ? fmtInt(trips.trips.length) : health.data ? fmtInt(health.data.n_trips) : 'N/A',
      sub: trips.status === 'ready' ? 'Listed by GET /trips' : health.data ? 'Reported by /health' : 'Loaded from the API',
    },
  ]

  return (
    <dl className="grid grid-cols-1 gap-2 sm:grid-cols-3">
      {cells.map(({ label, icon: Icon, tone, value, sub }) => (
        <div
          key={label}
          className={cn(
            'rounded-lg border bg-surface px-3.5 py-3',
            tone === 'ok' && 'border-hairline',
            tone === 'warn' && 'border-warn/45',
            tone === 'err' && 'border-err/45',
            tone === 'idle' && 'border-hairline',
          )}
        >
          <dt className="flex items-center justify-between gap-2 text-[12.5px] text-ink-muted">
            {label}
            <Icon className={cn('size-4', tone === 'ok' ? 'text-ok' : tone === 'warn' ? 'text-warn' : tone === 'err' ? 'text-err' : 'text-ink-faint')} />
          </dt>
          <dd className="mt-1.5 flex items-center gap-2">
            <StatusDot tone={tone === 'idle' ? 'busy' : tone} pulse={health.status === 'checking' && label.startsWith('API')} />
            <span className="num truncate text-[24px] leading-none font-semibold">{value}</span>
          </dd>
          <p className="mt-1.5 truncate text-[12px] text-ink-muted">{sub}</p>
        </div>
      ))}
    </dl>
  )
}

function LatestPrediction() {
  const { predictions, selectedTrip } = useSession()
  const latest = Object.values(predictions)
    .map((p) => p.result)
    .filter((r): r is PredictionResult => !!r)
    .sort((a, b) => b.receivedAt - a.receivedAt)[0]
  return (
    <p className="text-[13px] text-ink-muted">
      {latest ? (
        <>
          Last prediction: trip <span className="num text-[15px] font-semibold text-ink">{latest.response.trip_id}</span> at {fmtClock(latest.receivedAt)},{' '}
          {fmtMs(latest.response.timings_ms.total)} on the server.
        </>
      ) : selectedTrip ? (
        <>
          Trip <span className="num text-[15px] font-semibold text-ink">{selectedTrip.trip_id}</span> is selected and has not been predicted yet.
        </>
      ) : (
        'No predictions in this session yet.'
      )}
    </p>
  )
}

function PipelineFlow() {
  const { selectedTripId, predictions } = useSession()
  const result = selectedTripId ? predictions[selectedTripId]?.result : undefined
  const t = result?.response.timings_ms

  return (
    <Panel
      title="What happens when you run a prediction"
      description={
        t
          ? `Stage timings from the latest prediction for trip ${result!.response.trip_id}.`
          : 'Every stage runs on Laptop 1. Stage timings appear here after a prediction.'
      }
    >
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_auto_250px]">
        <div className="rounded-lg border border-hairline bg-inset p-3">
          <p className="mb-2.5 flex items-center gap-1.5 text-[12.5px] text-ink-muted">
            <Server className="size-3.5" /> Laptop 1: FastAPI server
          </p>
          <ol className="grid grid-cols-1 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            {STAGE_ORDER.map((k, i) => (
              <li key={k} className="relative flex flex-col rounded-md border border-hairline bg-surface px-3 py-2.5">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="grid size-5 place-items-center rounded-full bg-raised text-[11px] font-semibold text-ink-muted">{i + 1}</span>
                  <span className={cn('num text-[14px]', t?.[k] !== undefined ? 'font-semibold text-ink' : 'text-ink-faint')}>
                    {t?.[k] !== undefined ? fmtMs(t[k]) : t ? 'N/A' : ''}
                  </span>
                </div>
                <p className="mt-2 font-heading text-[16px] leading-tight font-semibold">{STAGE_LABELS[k]}</p>
                <p className="mt-1 text-[12px] leading-snug text-ink-muted">{STAGE_TEXT[k]}</p>
              </li>
            ))}
            <li className="flex flex-col rounded-md border border-dashed border-ok/45 bg-ok/[0.05] px-3 py-2.5">
              <span className="grid size-5 place-items-center rounded-full bg-ok/15 text-ok">
                <ShieldCheck className="size-3" />
              </span>
              <p className="mt-2 font-heading text-[16px] leading-tight font-semibold">Scoring</p>
              <p className="mt-1 text-[12px] leading-snug text-ink-muted">Reads the .route reference only after prediction, on the server.</p>
            </li>
          </ol>
          {t && <StageStrip timings={t} showLegend={false} className="mt-3" />}
        </div>

        <div className="hidden items-center justify-center xl:flex" aria-hidden="true">
          <div className="flex flex-col items-center gap-1 text-[11.5px] text-ink-faint">
            <ArrowRight className="size-5 text-route" />
            HTTP
          </div>
        </div>

        <div className="rounded-lg border border-route/35 bg-route/[0.05] p-3">
          <p className="mb-2 flex items-center gap-1.5 text-[12.5px] text-ink-muted">
            <MonitorSmartphone className="size-3.5" /> Laptop 2: this interface
          </p>
          <p className="text-[13px] font-medium text-ink">Receives only</p>
          <ul className="mt-1.5 space-y-1 text-[12.5px] text-ink-muted">
            {['GPS geometry', 'Predicted route geometry', 'Predicted arc per GPS point', 'Route arcs in travel order', 'Metrics and stage timings'].map((x) => (
              <li key={x} className="flex items-center gap-1.5">
                <span className="size-1.5 rounded-full bg-route" aria-hidden="true" /> {x}
              </li>
            ))}
          </ul>
          <p className="mt-2.5 border-t border-hairline pt-2 text-[12px] text-ink-faint">Never the reference route or its geometry.</p>
        </div>
      </div>
    </Panel>
  )
}

const PARTS = [
  { name: 'Training', n: 80, use: 'Model fitting, cross-validation and Bayesian optimisation', tone: 'var(--ps-series-1)' },
  { name: 'Student testing', n: 10, use: 'Development-stage evaluation and model comparison', tone: 'color-mix(in oklab, var(--ps-series-1) 55%, var(--ps-surface))' },
  { name: 'Evaluator testing', n: 10, use: 'Final evaluation only, run through this interface', tone: 'var(--ps-route)' },
]

export function Overview() {
  const { health } = useSession()

  return (
    <div className="flex flex-col gap-4">
      <section className="grid grid-cols-1 items-center gap-6 rounded-[14px] border border-hairline bg-surface/60 p-5 md:p-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        <div className="flex flex-col">
          <h2 className="max-w-[17ch] font-heading text-[36px] leading-[1.02] font-semibold tracking-[-0.005em] text-ink sm:text-[48px]">
            Noisy GPS in. A route on real roads out.
          </h2>
          <p className="mt-4 max-w-[58ch] text-[15px] leading-relaxed text-ink-muted">
            GPS fixes drift onto parallel streets, jump across buildings and drop out. PathSnap scores the candidate road arcs for every
            observation with a trained classifier, then decodes the most likely connected route with an HMM. This interface sends a trip
            to the API on Laptop 1 and shows exactly what comes back.
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            <Button asChild className="h-10 px-4 text-[14px] font-semibold">
              <Link to="/trips">
                <MapIcon className="size-4" /> Open trip explorer
              </Link>
            </Button>
            <Button asChild variant="outline" className="h-10 px-4 text-[14px]">
              <Link to="/evaluation">
                <ListChecks className="size-4" /> Evaluate all trips
              </Link>
            </Button>
          </div>
          <div className="mt-6">
            <StatusStrip />
            <div className="mt-2.5">
              {health.status === 'offline' ? (
                <p className="text-[13px] text-ink-muted">
                  The API is not reachable yet. Set Laptop 1&rsquo;s address from the status button in the top bar; trips load as soon as it answers.
                </p>
              ) : (
                <LatestPrediction />
              )}
            </div>
          </div>
        </div>
        <SnapSchematic />
      </section>

      <PipelineFlow />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <Panel title="Dataset partitions" description="From the PathSnap brief. Static project information, not read from the API.">
          <div className="flex h-3.5 gap-[2px] overflow-hidden rounded-[3px]" role="img" aria-label="80 training, 10 student testing and 10 evaluator testing trips">
            {PARTS.map((p) => (
              <span key={p.name} style={{ flexGrow: p.n, background: p.tone }} />
            ))}
          </div>
          <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
            {PARTS.map((p) => (
              <li key={p.name} className="rounded-lg border border-hairline bg-raised/40 px-3 py-2.5">
                <p className="flex items-center gap-1.5 text-[12.5px] text-ink-muted">
                  <span className="size-2.5 rounded-[2px]" style={{ background: p.tone }} aria-hidden="true" /> {p.name}
                </p>
                <p className="num mt-1 text-[28px] leading-none font-semibold">
                  {p.n} <span className="font-sans text-[13px] font-normal text-ink-faint">trips</span>
                </p>
                <p className="mt-1.5 text-[12px] leading-snug text-ink-muted">{p.use}</p>
              </li>
            ))}
          </ul>
          <p className="mt-2.5 text-[12px] text-ink-faint">
            No trip is split across partitions. The trips shown in this interface always come from GET /trips, never from this table.
          </p>
        </Panel>

        <Panel title="What this interface can see" description="Evaluation integrity rules from the brief, enforced in the UI.">
          <ul className="space-y-3 text-[13.5px]">
            {[
              ['Reference routes never leave Laptop 1.', 'The API scores predictions on the server and returns metrics only. Any map layer named like a reference route is refused.'],
              ['Nothing is hardcoded.', 'Trip ids, routes and metrics are fetched from the API on every run.'],
              ['Missing means N/A.', 'A null metric is shown as N/A, never as zero, and stub output is labelled as a placeholder.'],
            ].map(([a, b]) => (
              <li key={a} className="flex gap-2.5">
                <ShieldCheck className="mt-0.5 size-4 shrink-0 text-ok" />
                <span>
                  <span className="font-medium text-ink">{a}</span> <span className="text-ink-muted">{b}</span>
                </span>
              </li>
            ))}
          </ul>
          <a
            href="https://github.com/Ashrit-Ayush-Bhuvi-Chinmay/ML-HACKATHON"
            target="_blank"
            rel="noreferrer"
            className="mt-4 inline-flex items-center gap-1 text-[13px] text-ink-muted underline-offset-4 hover:text-ink hover:underline"
          >
            Project repository <ArrowUpRight className="size-3.5" />
          </a>
        </Panel>
      </div>
    </div>
  )
}
