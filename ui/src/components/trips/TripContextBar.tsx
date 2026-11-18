import { NavLink } from 'react-router'
import { ChartColumn, ChevronDown, ChevronLeft, ChevronRight, Map as MapIcon, Play, Route } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { fmtInt } from '@/lib/format'
import { viewOf } from '@/lib/predictionView'
import { cn } from '@/lib/utils'
import { useSession } from '@/state/session'
import { StatusPill } from './PredictionPanel'
import { tripStatus } from './TripList'

const VIEWS = [
  { to: '/trips', label: 'Map', icon: MapIcon },
  { to: '/route', label: 'Route analysis', icon: Route },
  { to: '/analytics', label: 'Analytics', icon: ChartColumn },
]

const STATUS_WORD = { 'not-run': 'not run', running: 'predicting', done: 'predicted', failed: 'failed' } as const

/** Keeps the selected trip in view across pages: switch trip, run it, jump between its views. */
export function TripContextBar() {
  const { trips, selectedTripId, selectTrip, predictions, runPrediction, health, batch } = useSession()
  const list = trips.trips
  const idx = list.findIndex((t) => t.trip_id === selectedTripId)
  const rec = selectedTripId ? predictions[selectedTripId] : undefined
  const placeholder = rec?.result ? viewOf(rec.result.response).placeholder : false
  const canRun = health.status === 'online' && !batch.running && rec?.status !== 'running'

  if (list.length === 0) return null

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-[10px] border border-hairline bg-surface px-3 py-2">
      <div className="flex items-center gap-1">
        <Button
          size="icon-sm"
          variant="ghost"
          aria-label="Previous trip"
          disabled={idx <= 0}
          onClick={() => selectTrip(list[idx - 1].trip_id)}
        >
          <ChevronLeft />
        </Button>
        <label className="relative">
          <span className="sr-only">Trip</span>
          <select
            value={selectedTripId ?? ''}
            onChange={(e) => selectTrip(e.target.value || null)}
            className="num h-8 appearance-none rounded-md border border-hairline bg-inset pr-8 pl-2.5 text-[16px] font-semibold text-ink focus:border-route/60 focus:outline-none"
          >
            {!selectedTripId && <option value="">Choose a trip</option>}
            {list.map((t) => (
              <option key={t.trip_id} value={t.trip_id}>
                {t.trip_id} ({STATUS_WORD[tripStatus(predictions[t.trip_id])]})
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute top-1/2 right-2 size-4 -translate-y-1/2 text-ink-faint" />
        </label>
        <Button
          size="icon-sm"
          variant="ghost"
          aria-label="Next trip"
          disabled={idx === -1 || idx >= list.length - 1}
          onClick={() => selectTrip(list[idx + 1].trip_id)}
        >
          <ChevronRight />
        </Button>
      </div>

      {selectedTripId && (
        <>
          <StatusPill rec={rec} placeholder={placeholder} />
          <span className="text-[12.5px] text-ink-muted">{fmtInt(list[idx]?.n_points)} GPS points</span>
          {!rec?.result && (
            <Button size="sm" onClick={() => void runPrediction(selectedTripId)} disabled={!canRun}>
              <Play className="size-3.5" /> Run prediction
            </Button>
          )}
        </>
      )}

      <nav aria-label="Views of this trip" className="ml-auto flex items-center gap-1">
        {VIEWS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'flex h-8 items-center gap-1.5 rounded-md px-2.5 text-[13px] transition-colors',
                isActive ? 'bg-raised text-ink' : 'text-ink-muted hover:bg-raised/60 hover:text-ink',
              )
            }
          >
            <Icon className="size-3.5" /> <span className="hidden sm:inline">{label}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  )
}
