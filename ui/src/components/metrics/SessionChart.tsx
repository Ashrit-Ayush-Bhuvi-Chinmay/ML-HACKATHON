import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmtPct } from '@/lib/format'

export interface SessionRow {
  trip_id: string
  top1: number | null
  sequenceF1: number | null
}

const AXIS = { fill: 'var(--ps-ink-faint)', fontSize: 12 }

/** Top-1 accuracy and sequence F1 per trip, as returned by the API during this session. */
export function SessionChart({ rows }: { rows: SessionRow[] }) {
  return (
    <div className="h-[280px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -8 }} barGap={2} barCategoryGap="28%">
          <CartesianGrid vertical={false} stroke="var(--ps-hairline)" strokeDasharray="2 4" />
          <XAxis dataKey="trip_id" tick={AXIS} axisLine={{ stroke: 'var(--ps-hairline)' }} tickLine={false} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(v: number) => `${v * 100}%`} tick={AXIS} axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: 'var(--ps-raised)', opacity: 0.5 }}
            contentStyle={{ background: 'var(--ps-raised)', border: '1px solid var(--ps-hairline)', borderRadius: 6, fontSize: 12 }}
            labelStyle={{ color: 'var(--ps-ink)', marginBottom: 2 }}
            labelFormatter={(l) => `Trip ${l}`}
            formatter={(v, name) => [v === null || v === undefined ? 'N/A' : fmtPct(Number(v)), name]}
          />
          <Legend
            verticalAlign="top"
            itemSorter={null}
            align="right"
            height={28}
            iconType="square"
            iconSize={10}
            formatter={(v) => <span style={{ color: 'var(--ps-ink-muted)', fontSize: 12.5 }}>{v}</span>}
          />
          <Bar dataKey="top1" name="Top-1 accuracy" fill="var(--ps-series-1)" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false} />
          <Bar dataKey="sequenceF1" name="Sequence F1" fill="var(--ps-series-2)" radius={[4, 4, 0, 0]} maxBarSize={22} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
