import { CircleX, Loader2, MapIcon, Play, RotateCw, Unplug } from 'lucide-react'
import { EmptyState } from '@/components/common/EmptyState'
import { RouteMap } from '@/components/map/RouteMap'
import { MatchRibbon } from '@/components/ribbon/MatchRibbon'
import { PredictionPanel } from '@/components/trips/PredictionPanel'
import { TripList } from '@/components/trips/TripList'
import { Button } from '@/components/ui/button'
import { useNow } from '@/hooks/useNow'
import { fmtInt } from '@/lib/format'
import { useSession } from '@/state/session'

function MapOverlay() {
  const { selectedTrip, predictions, runPrediction, health, batch } = useSession()
  const rec = selectedTrip ? predictions[selectedTrip.trip_id] : undefined
  const running = rec?.status === 'running'
  const now = useNow(running, 100)

  if (!selectedTrip) {
    return (
      <EmptyState icon={MapIcon} title="Choose a trip to begin" className="rounded-xl border border-hairline bg-surface/95">
        Select an evaluation trip from the list. Its GPS track and the reconstructed route appear here after the prediction runs.
      </EmptyState>
    )
  }
  if (rec?.result) return null
  if (!running && health.status === 'offline') {
    return (
      <EmptyState icon={Unplug} title="The API is offline" className="rounded-xl border border-hairline bg-surface/95">
        Trip {selectedTrip.trip_id} can be predicted once Laptop 1&rsquo;s API answers. Check the address from the status button in the top bar.
      </EmptyState>
    )
  }
  if (running) {
    return (
      <div className="flex w-[min(380px,100%)] flex-col items-center rounded-xl border border-route/35 bg-surface/95 px-8 py-7 text-center" aria-live="polite">
        <Loader2 className="size-7 animate-spin text-route" />
        <p className="mt-3 font-heading text-[20px] font-semibold">Predicting trip {selectedTrip.trip_id}</p>
        <p className="mt-1 text-[13px] text-ink-muted">
          The API is matching {fmtInt(selectedTrip.n_points)} GPS points to the road network.
        </p>
        <p className="num mt-3 text-[28px] leading-none font-semibold text-ink">{(Math.max(0, now - rec.startedAt) / 1000).toFixed(1)} s</p>
        <div className="mt-4 h-1 w-full overflow-hidden rounded-full bg-hairline">
          <div className="h-full w-1/3 animate-[ps-indeterminate_1.4s_ease-in-out_infinite] rounded-full bg-route" />
        </div>
      </div>
    )
  }
  if (rec?.status === 'error' && rec.error && rec.error.kind !== 'cancelled') {
    return (
      <EmptyState
        icon={CircleX}
        title={`Prediction failed for trip ${selectedTrip.trip_id}`}
        className="rounded-xl border border-err/40 bg-surface/95"
        action={
          health.status === 'online' && (
            <Button variant="outline" onClick={() => void runPrediction(selectedTrip.trip_id)}>
              <RotateCw /> Try again
            </Button>
          )
        }
      >
        The API did not return a route. The reason is shown in the prediction panel.
      </EmptyState>
    )
  }
  return (
    <EmptyState
      icon={MapIcon}
      title={`Trip ${selectedTrip.trip_id} is ready`}
      className="rounded-xl border border-hairline bg-surface/95"
      action={
        <Button
          onClick={() => void runPrediction(selectedTrip.trip_id)}
          disabled={health.status !== 'online' || batch.running}
          className="h-9 px-4 font-semibold"
        >
          <Play className="size-4" /> Run prediction
        </Button>
      }
    >
      {fmtInt(selectedTrip.n_points)} GPS points. The track and the predicted route load from the API when you run the prediction.
    </EmptyState>
  )
}

export function TripExplorer() {
  const { selectedTripId, predictions } = useSession()
  const result = selectedTripId ? predictions[selectedTripId]?.result : undefined

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:h-[calc(100dvh-6rem)] lg:flex-none lg:grid-cols-[244px_minmax(0,1fr)_304px] min-[1600px]:grid-cols-[264px_minmax(0,1fr)_330px] lg:grid-rows-[minmax(0,1fr)]">
      <TripList className="max-h-[360px] lg:max-h-none" />

      <div className="flex min-h-0 flex-col gap-3">
        <RouteMap result={result} overlay={result ? undefined : <MapOverlay />} className="h-[62vh] lg:h-auto lg:flex-1" />
        {result && (
          <section className="rounded-[10px] border border-hairline bg-surface px-4 pt-3 pb-2" aria-label="Match ribbon">
            <MatchRibbon arcs={result.response.predicted_arcs} height={20} showAxis={false} />
          </section>
        )}
      </div>

      <aside className="scroll-thin min-h-0 overflow-y-auto rounded-[10px] border border-hairline bg-surface" aria-label="Prediction">
        <PredictionPanel />
      </aside>
    </div>
  )
}
