import type { TimingsMs } from '@/api/types'
import { InfoTip } from '@/components/common/InfoTip'
import { fmtMs, stagesOf, STAGE_ORDER } from '@/lib/format'
import { StageStrip } from './StageStrip'

const STAGE_HINT: Record<string, string> = {
  load: 'Reading the trip’s .track, .nodes and .arcs files on Laptop 1.',
  candidates: 'Finding the road arcs each GPS point could belong to.',
  features: 'Building the classifier’s input features for each point–arc pair.',
  classifier: 'Scoring the candidates with the selected ML model.',
  hmm: 'HMM / Viterbi decoding of the most likely connected route.',
}

/**
 * Server stage timings from timings_ms as a readable table: a bar per stage on a shared scale,
 * the time, and its share of the server total. Stages the API did not report are listed as N/A.
 */
export function TimingChart({ timings, roundTripMs }: { timings: TimingsMs; roundTripMs?: number }) {
  const { stages, total, unattributed } = stagesOf(timings)
  const reported = new Map(stages.map((s) => [s.key, s]))
  const rows = [
    ...STAGE_ORDER.map((k) => ({ key: k, label: reported.get(k)?.label ?? k, ms: reported.get(k)?.ms ?? null, hint: STAGE_HINT[k] })),
    ...stages.filter((s) => !(STAGE_ORDER as readonly string[]).includes(s.key)).map((s) => ({ key: s.key, label: s.label, ms: s.ms, hint: 'Extra stage reported by the API.' })),
    ...(unattributed !== null ? [{ key: 'other', label: 'Other', ms: unattributed, hint: 'Server total minus the listed stages.' }] : []),
  ]
  const labels: Record<string, string> = { load: 'Load trip', candidates: 'Candidate arcs', features: 'Features', classifier: 'Classifier', hmm: 'HMM / Viterbi' }
  const max = Math.max(1, ...rows.map((r) => r.ms ?? 0))

  return (
    <div>
      <div className="mb-3 grid grid-cols-2 gap-2">
        <div className="rounded-lg border border-hairline bg-raised/45 px-3 py-2.5">
          <p className="flex items-center gap-1 text-[12.5px] text-ink-muted">
            Server total <InfoTip label="About server total">timings_ms.total as reported by the API.</InfoTip>
          </p>
          <p className="num mt-1 text-[26px] leading-none font-semibold">{fmtMs(total)}</p>
        </div>
        <div className="rounded-lg border border-hairline bg-raised/45 px-3 py-2.5">
          <p className="flex items-center gap-1 text-[12.5px] text-ink-muted">
            Round trip <InfoTip label="About round trip">Measured in this browser from sending the request to parsing the response, so it includes network time.</InfoTip>
          </p>
          <p className="num mt-1 text-[26px] leading-none font-semibold">{fmtMs(roundTripMs)}</p>
        </div>
      </div>

      {stages.length > 0 && <StageStrip timings={timings} showLegend={false} className="mb-3" />}

      <table className="w-full text-[13px]">
        <caption className="sr-only">Server stage timings</caption>
        <thead className="text-left text-[12px] text-ink-faint">
          <tr className="border-b border-hairline">
            <th className="py-1.5 pr-2 font-medium">Stage</th>
            <th className="w-[44%] px-2 py-1.5 font-medium">
              <span className="sr-only">Bar</span>
            </th>
            <th className="px-2 py-1.5 text-right font-medium">Time</th>
            <th className="py-1.5 pl-2 text-right font-medium">Share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-b border-hairline/50 last:border-0">
              <td className="py-1.5 pr-2 whitespace-nowrap">
                <span className="inline-flex items-center gap-1 text-ink">
                  {labels[r.key] ?? r.label}
                  <InfoTip label={`About ${labels[r.key] ?? r.label}`}>{r.hint}</InfoTip>
                </span>
              </td>
              <td className="px-2 py-1.5">
                {r.ms !== null ? (
                  <div className="h-2 rounded-full bg-hairline/60">
                    <div
                      className="h-full min-w-[2px] rounded-full"
                      style={{ width: `${(r.ms / max) * 100}%`, background: r.key === 'other' ? 'var(--ps-ink-faint)' : 'var(--ps-series-1)' }}
                    />
                  </div>
                ) : (
                  <div className="h-2 rounded-full border border-dashed border-hairline" title="Not reported" />
                )}
              </td>
              <td className={r.ms === null ? 'px-2 py-1.5 text-right text-ink-faint' : 'px-2 py-1.5 text-right text-ink'}>{r.ms === null ? 'N/A' : fmtMs(r.ms)}</td>
              <td className="py-1.5 pl-2 text-right text-ink-muted">{r.ms !== null && total ? `${Math.round((r.ms / total) * 100)}%` : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {stages.length === 0 && (
        <p className="mt-2 text-[12.5px] text-ink-faint">The API reported only the total time. Stage rows fill in when the pipeline returns per-stage timings.</p>
      )}
    </div>
  )
}
