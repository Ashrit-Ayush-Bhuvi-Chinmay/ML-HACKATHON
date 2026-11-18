import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type L from 'leaflet'
import { Crosshair, Expand, Layers, MapPinOff, Minus, Play, Plus, Shrink } from 'lucide-react'
import { GeoJSON, MapContainer, TileLayer } from 'react-leaflet'
import { Segmented } from '@/components/common/Segmented'
import { PointDetails } from '@/components/ribbon/PointDetails'
import type { PredictOptions } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { TILE_ATTRIBUTION, TILE_DARKEN, TILE_URL } from '@/config'
import { routeDifference, segmentSet, type MapLayer } from '@/lib/geo'
import { fmtInt, fmtPct } from '@/lib/format'
import { viewOf } from '@/lib/predictionView'
import { cn } from '@/lib/utils'
import { runKeyOf, runLabelOf, useSession, type PredictionResult } from '@/state/session'
import { KIND_COLOR, MAP_COLORS } from './layerStyle'
import {
  AutoResize,
  FitBounds,
  FocusMarker,
  GpsPointsLayer,
  HighlightLayer,
  MapPanes,
  PANES,
  PanToPoint,
  ScaleAndCoords,
  TripEndpoints,
} from './MapLayers'

type Basemap = 'dark' | 'light' | 'off'
const BASEMAP_KEY = 'pathsnap.basemap'

/** Muted, distinct colours for overlaid comparison routes. */
const OVERLAY_COLORS = ['#f0a35e', '#c084fc', '#93c5fd', '#f9a8d4', '#a3e635', '#fcd34d']
const NO_RUNS: PredictionResult[] = []
const NO_KEYS: string[] = []

const QUICK_STEPS: { options: PredictOptions; label: string }[] = [
  { options: { model: 'distance_gaussian', hmm: false }, label: 'distance_gaussian, no HMM' },
  { options: { model: 'default', hmm: false }, label: 'default, classifier only' },
  { options: { model: 'default', hmm: true }, label: 'default, HMM' },
]

/** Positions where two predicted_arcs arrays disagree; null counts as a value. */
function differingPoints(a: (number | null)[], b: (number | null)[]) {
  const total = Math.max(a.length, b.length)
  let diff = 0
  for (let i = 0; i < total; i++) if ((a[i] ?? null) !== (b[i] ?? null)) diff++
  return { diff, total }
}

function readBasemap(): Basemap {
  try {
    const v = window.localStorage.getItem(BASEMAP_KEY)
    if (v === 'dark' || v === 'light' || v === 'off') return v
  } catch {
    /* ignore */
  }
  return 'dark'
}

function LineSwatch({ color, width, dashed }: { color: string; width: number; dashed?: boolean }) {
  return (
    <svg width="22" height="10" aria-hidden="true" className="shrink-0">
      {width > 3 && <line x1="2" y1="5" x2="20" y2="5" stroke={MAP_COLORS.routeCasing} strokeWidth={width + 3} strokeLinecap="round" />}
      <line x1="2" y1="5" x2="20" y2="5" stroke={color} strokeWidth={width} strokeLinecap="round" strokeDasharray={dashed ? '4 3' : undefined} />
    </svg>
  )
}

function DotSwatch({ hollow }: { hollow?: boolean }) {
  return (
    <span className="flex w-[22px] shrink-0 items-center justify-center gap-[3px]" aria-hidden="true">
      {[0, 1].map((i) => (
        <span key={i} className={cn('size-[6px] rounded-full', hollow ? 'border border-unmatched' : 'bg-gps')} />
      ))}
    </span>
  )
}

function LayerRow({
  id,
  swatch,
  label,
  sub,
  checked,
  onChange,
}: {
  id: string
  swatch: ReactNode
  label: string
  sub: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <li className="flex items-center gap-2.5">
      {swatch}
      <label htmlFor={id} className="min-w-0 flex-1 cursor-pointer">
        <span className="block truncate leading-tight text-ink">{label}</span>
        <span className="block truncate text-[11.5px] leading-tight text-ink-faint">{sub}</span>
      </label>
      <Switch id={id} size="sm" checked={checked} onCheckedChange={onChange} />
    </li>
  )
}

function ControlButton({ label, onClick, children, disabled }: { label: string; onClick: () => void; children: ReactNode; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="grid size-8 place-items-center text-ink-muted transition-colors hover:bg-raised hover:text-ink disabled:opacity-40"
    >
      {children}
    </button>
  )
}

export function RouteMap({
  result,
  overlay,
  highlight,
  className,
  compactLegend = false,
  showPointCard = true,
}: {
  result?: PredictionResult
  /** Content shown over the map when there is no result (empty / loading / error). */
  overlay?: ReactNode
  highlight?: { start: number; end: number } | null
  className?: string
  compactLegend?: boolean
  /** Show the focused-point card over the map (off where the page has its own inspector). */
  showPointCard?: boolean
}) {
  const { focusIndex, setFocusIndex, runsForTrip, findStoredRun, runPrediction, health, batch, overlayKeysFor, setOverlayKeysFor } = useSession()
  const response = result?.response
  const view = useMemo(() => (response ? viewOf(response) : null), [response])
  const placeholder = view?.placeholder ?? false

  const wrapRef = useRef<HTMLDivElement>(null)
  const [map, setMap] = useState<L.Map | null>(null)
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  const [basemap, setBasemapState] = useState<Basemap>(readBasemap)
  const [tileErrors, setTileErrors] = useState(0)
  const [fitNonce, setFitNonce] = useState(0)
  // Start collapsed on phones so the panel does not cover the map.
  const [legendOpen, setLegendOpen] = useState(() => !compactLegend && window.innerWidth >= 768)
  const [fullscreen, setFullscreen] = useState(false)
  const [compareOpen, setCompareOpen] = useState(true)
  const [quick, setQuick] = useState<{ tripId: string; running: boolean; done: number; total: number; label: string; error?: string } | null>(null)

  const currentTripId = response?.trip_id ?? null
  const compareKeys = currentTripId ? overlayKeysFor(currentTripId) : NO_KEYS
  const setCompareKeys = (keys: string[] | ((prev: string[]) => string[])) => {
    if (!currentTripId) return
    const next = typeof keys === 'function' ? keys(compareKeys) : keys
    setOverlayKeysFor(currentTripId, next)
  }

  const tripRuns = response ? runsForTrip(response.trip_id) : NO_RUNS
  const currentKey = result ? runKeyOf(result) : null
  const otherRuns = tripRuns.filter((run) => runKeyOf(run) !== currentKey)
  const selectedRuns = otherRuns.filter((run) => compareKeys.includes(runKeyOf(run)))
  const tripIdRef = useRef(currentTripId)
  useEffect(() => {
    tripIdRef.current = currentTripId
  }, [currentTripId])

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === wrapRef.current)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  const setBasemap = (b: Basemap) => {
    setBasemapState(b)
    setTileErrors(0)
    try {
      window.localStorage.setItem(BASEMAP_KEY, b)
    } catch {
      /* ignore */
    }
  }

  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen()
    else void wrapRef.current?.requestFullscreen?.()
  }

  const resultKey = response ? `${response.trip_id}-${result!.receivedAt}` : 'none'
  const gpsPoints = view?.layers.gpsPoints ?? []
  const arcs = response?.predicted_arcs ?? []
  const focusPoint = focusIndex !== null ? (gpsPoints[focusIndex] ?? null) : null
  const focusRun = focusIndex !== null && view ? view.pointInfo(focusIndex).run : null
  const runHighlight = highlight ?? (focusRun && focusRun.length > 1 ? { start: focusRun.start, end: focusRun.end } : null)

  const layers = view?.layers.layers ?? []
  const lineLayers = layers.filter((l) => l.kind !== 'gps')
  const hasGps = layers.some((l) => l.kind === 'gps')
  const routeLayer = layers.find((l) => l.kind === 'route')
  const extraLayers = layers.filter((l) => l.kind === 'extra')
  const roadLayer = layers.find((l) => l.kind === 'road')
  const stats = view?.stats
  // Segments of the current route, to highlight only where an overlaid run goes elsewhere.
  const baseSegments = useMemo(() => (routeLayer ? segmentSet(routeLayer.data) : new Set<string>()), [routeLayer])

  const toggle = (key: string) => (v: boolean) => setHidden((h) => ({ ...h, [key]: !v }))

  const colorFor = (key: string) => {
    const index = otherRuns.findIndex((run) => runKeyOf(run) === key)
    return OVERLAY_COLORS[(index < 0 ? 0 : index) % OVERLAY_COLORS.length]
  }

  const toggleCompare = (key: string, on: boolean) =>
    setCompareKeys((keys) => (on ? [...new Set([...keys, key])] : keys.filter((k) => k !== key)))

  const runQuickComparison = async () => {
    if (!response || quick?.running) return
    const tripId = response.trip_id
    setQuick({ tripId, running: true, done: 0, total: QUICK_STEPS.length, label: QUICK_STEPS[0].label })
    const added: string[] = []
    for (let i = 0; i < QUICK_STEPS.length; i++) {
      const step = QUICK_STEPS[i]
      setQuick((q) => (q ? { ...q, label: step.label } : q))
      const outcome = await runPrediction(tripId, step.options, true, false)
      if (tripIdRef.current !== tripId) return
      if (outcome !== 'success') {
        setQuick((q) => (q ? { ...q, running: false, error: 'A run failed. See the prediction panel for the reason.' } : q))
        return
      }
      const stored = findStoredRun(tripId, step.options)
      if (stored) {
        added.push(runKeyOf(stored))
        setCompareKeys([...added])
      }
      setQuick((q) => (q ? { ...q, done: i + 1 } : q))
    }
    setCompareKeys(added)
    setCompareOpen(true)
    setQuick((q) => (q ? { ...q, running: false, label: 'Done' } : q))
  }

  return (
    <div
      ref={wrapRef}
      className={cn('relative isolate min-h-[320px] overflow-hidden rounded-[10px] border border-hairline bg-inset', fullscreen && 'rounded-none border-0', className)}
    >
      <MapContainer
        ref={setMap}
        center={[28, 12]}
        zoom={2}
        minZoom={2}
        maxZoom={19}
        preferCanvas
        worldCopyJump
        zoomControl={false}
        className="absolute inset-0 h-full w-full"
      >
        <MapPanes />
        <AutoResize />
        <ScaleAndCoords />
        {basemap !== 'off' && (
          <TileLayer
            key={basemap}
            url={TILE_URL}
            attribution={TILE_ATTRIBUTION}
            className={basemap === 'dark' && TILE_DARKEN ? 'ps-tiles-dark' : undefined}
            maxZoom={19}
            eventHandlers={{ tileerror: () => setTileErrors((n) => n + 1), tileload: () => setTileErrors(0) }}
          />
        )}

        {lineLayers.map((layer) =>
          hidden[layer.key] ? null : layer.kind === 'route' ? (
            <RouteLine key={`${resultKey}-${layer.key}`} layer={layer} placeholder={placeholder} />
          ) : (
            <GeoJSON
              key={`${resultKey}-${layer.key}`}
              data={layer.data as GeoJSON.FeatureCollection}
              interactive={false}
              pane={layer.kind === 'road' ? PANES.road : PANES.extra}
              style={() => ({
                color: KIND_COLOR[layer.kind],
                weight: layer.kind === 'road' ? 1.3 : 2.5,
                opacity: layer.kind === 'road' ? 0.8 : 0.9,
              })}
            />
          ),
        )}

        {/* Earlier runs of the same trip, translucent and below the current route. */}
        {selectedRuns.map((run) => {
          const layer = viewOf(run.response).layers.layers.find((l) => l.kind === 'route')
          if (!layer) return null
          const key = runKeyOf(run)
          const diff = routeDifference(layer.data, baseSegments)
          return (
            <Fragment key={`compare-wrap-${key}`}>
              <GeoJSON
                key={`compare-${key}`}
                data={layer.data as GeoJSON.FeatureCollection}
                interactive={false}
                pane={PANES.overlay}
                style={() => ({ color: colorFor(key), weight: 3, opacity: 0.45, dashArray: '6 5', lineCap: 'round', lineJoin: 'round' })}
                onEachFeature={(_f, l) => l.bindTooltip(runLabelOf(run), { sticky: true, className: 'ps-tooltip' })}
              />
              {diff.segments > 0 && (
                <GeoJSON
                  key={`compare-diff-${key}-${diff.segments}`}
                  data={diff.data as GeoJSON.FeatureCollection}
                  pane={PANES.highlight}
                  style={() => ({ color: colorFor(key), weight: 5, opacity: 0.95, lineCap: 'round', lineJoin: 'round' })}
                  onEachFeature={(_f, l) =>
                    l.bindTooltip(`${runLabelOf(run)}: goes where the current route does not`, { sticky: true, className: 'ps-tooltip' })
                  }
                />
              )}
            </Fragment>
          )
        })}

        {hasGps && (
          <GpsPointsLayer
            key={`${resultKey}-gps`}
            points={gpsPoints}
            arcs={arcs}
            showMatched={placeholder ? !hidden['gps:all'] : !hidden['gps:matched']}
            showUnmatched={placeholder ? !hidden['gps:all'] : !hidden['gps:unmatched']}
            uniform={placeholder}
            onPick={(i) => setFocusIndex(i)}
          />
        )}
        <HighlightLayer points={gpsPoints} range={runHighlight} />
        {hasGps && !hidden.endpoints && <TripEndpoints key={`${resultKey}-ends`} points={gpsPoints} />}
        <FocusMarker point={focusPoint} />
        <FitBounds bounds={view?.bounds ?? null} fitKey={`${resultKey}-${fitNonce}`} reserveRight={legendOpen} />
        <PanToPoint point={focusPoint} />
      </MapContainer>

      {/* Map controls */}
      <div className="absolute top-3 left-3 z-[500] flex flex-col overflow-hidden rounded-lg border border-hairline bg-surface/95 shadow-[0_6px_18px_rgb(0_0_0/0.3)] backdrop-blur">
        <ControlButton label="Zoom in" onClick={() => map?.zoomIn()}>
          <Plus className="size-4" />
        </ControlButton>
        <ControlButton label="Zoom out" onClick={() => map?.zoomOut()}>
          <Minus className="size-4" />
        </ControlButton>
        <span className="h-px bg-hairline" />
        <ControlButton label="Fit the trip in view" onClick={() => setFitNonce((n) => n + 1)} disabled={!view?.bounds}>
          <Crosshair className="size-4" />
        </ControlButton>
        <ControlButton label={fullscreen ? 'Exit full screen' : 'Full screen map'} onClick={toggleFullscreen}>
          {fullscreen ? <Shrink className="size-4" /> : <Expand className="size-4" />}
        </ControlButton>
      </div>

      {/* Right-hand map panels: layer legend and run comparison */}
      {response && (
        <div className="absolute top-3 right-3 z-[500] flex max-h-[calc(100%-1.5rem)] w-[248px] max-w-[calc(100%-4.5rem)] flex-col gap-2 overflow-y-auto">
        <div className="rounded-lg border border-hairline bg-surface/95 text-[13px] shadow-[0_8px_24px_rgb(0_0_0/0.35)] backdrop-blur">
          <button
            type="button"
            className="flex w-full items-center justify-between gap-2 px-3 py-2"
            onClick={() => setLegendOpen((o) => !o)}
            aria-expanded={legendOpen}
          >
            <span className="flex items-center gap-1.5 font-medium">
              <Layers className="size-3.5 text-ink-muted" /> Map layers
            </span>
            <span className="text-[12px] text-ink-faint">{legendOpen ? 'Hide' : 'Show'}</span>
          </button>
          {legendOpen && (
            <div className="border-t border-hairline px-3 pt-2.5 pb-3">
              <ul className="space-y-2">
                {routeLayer && (
                  <LayerRow
                    id="layer-route"
                    swatch={
                      placeholder ? <LineSwatch color={MAP_COLORS.placeholder} width={2} dashed /> : <LineSwatch color={MAP_COLORS.route} width={4} />
                    }
                    label={placeholder ? 'Placeholder line' : 'Predicted route'}
                    sub={placeholder ? 'Raw GPS joined up, not a match' : `${fmtInt(response.route_arcs.length)} arcs · ${fmtInt(response.n_route_pieces ?? 1)} pieces`}
                    checked={!hidden[routeLayer.key]}
                    onChange={toggle(routeLayer.key)}
                  />
                )}
                {hasGps && stats && placeholder && (
                  <LayerRow
                    id="layer-gps-all"
                    swatch={<DotSwatch />}
                    label="GPS observations"
                    sub={`${fmtInt(stats.total)} points, none matched`}
                    checked={!hidden['gps:all']}
                    onChange={toggle('gps:all')}
                  />
                )}
                {hasGps && stats && !placeholder && (
                  <>
                    <LayerRow
                      id="layer-gps-matched"
                      swatch={<DotSwatch />}
                      label="Matched GPS points"
                      sub={`${fmtInt(stats.matched)} of ${fmtInt(stats.total)} have an arc`}
                      checked={!hidden['gps:matched']}
                      onChange={toggle('gps:matched')}
                    />
                    <LayerRow
                      id="layer-gps-unmatched"
                      swatch={<DotSwatch hollow />}
                      label="Unmatched GPS points"
                      sub={`${fmtInt(stats.unmatched)} with no arc`}
                      checked={!hidden['gps:unmatched']}
                      onChange={toggle('gps:unmatched')}
                    />
                  </>
                )}
                {roadLayer ? (
                  <LayerRow
                    id="layer-road"
                    swatch={<LineSwatch color={MAP_COLORS.road} width={1.5} />}
                    label="Road network"
                    sub={`${fmtInt(roadLayer.vertexCount)} vertices`}
                    checked={!hidden[roadLayer.key]}
                    onChange={toggle(roadLayer.key)}
                  />
                ) : (
                  <li className="flex items-center gap-2.5 text-ink-faint">
                    <span className="flex w-[22px] justify-center">
                      <MapPinOff className="size-3.5" />
                    </span>
                    <span className="text-[12px] leading-tight">Road network not included in this response</span>
                  </li>
                )}
                {extraLayers.map((l) => (
                  <LayerRow
                    key={l.key}
                    id={`layer-${l.key}`}
                    swatch={<LineSwatch color={MAP_COLORS.extra} width={2.5} />}
                    label={l.label}
                    sub={`${fmtInt(l.vertexCount)} vertices, sent by the API`}
                    checked={!hidden[l.key]}
                    onChange={toggle(l.key)}
                  />
                ))}
                {hasGps && (
                  <LayerRow
                    id="layer-endpoints"
                    swatch={
                      <span className="flex w-[22px] justify-center" aria-hidden="true">
                        <span className="ps-endpoint ps-endpoint-static">S</span>
                      </span>
                    }
                    label="Start and end"
                    sub="First and last GPS point"
                    checked={!hidden.endpoints}
                    onChange={toggle('endpoints')}
                  />
                )}
              </ul>
              <div className="mt-3 flex items-center justify-between gap-2 border-t border-hairline pt-2.5">
                <span className="text-ink-muted">Basemap</span>
                <Segmented<Basemap>
                  label="Basemap style"
                  size="xs"
                  value={basemap}
                  onChange={setBasemap}
                  options={
                    TILE_DARKEN
                      ? [
                          { value: 'dark', label: 'Dark' },
                          { value: 'light', label: 'Light' },
                          { value: 'off', label: 'Off' },
                        ]
                      : [
                          { value: 'dark', label: 'On' },
                          { value: 'off', label: 'Off' },
                        ]
                  }
                />
              </div>
            </div>
          )}
        </div>

        {otherRuns.length > 0 && (
          <div className="rounded-lg border border-hairline bg-surface/95 text-[13px] shadow-[0_8px_24px_rgb(0_0_0/0.35)] backdrop-blur">
            <button
              type="button"
              className="flex w-full items-center justify-between gap-2 px-3 py-2"
              onClick={() => setCompareOpen((o) => !o)}
              aria-expanded={compareOpen}
            >
              <span className="font-medium">Compare runs</span>
              <span className="text-[12px] text-ink-faint">{compareOpen ? 'Hide' : `${otherRuns.length} stored`}</span>
            </button>
            {compareOpen && (
              <div className="border-t border-hairline px-3 pt-2.5 pb-3">
                <div className="scroll-thin max-h-[26vh] space-y-1 overflow-y-auto pr-1">
                  {otherRuns.map((run) => {
                    const key = runKeyOf(run)
                    const checked = compareKeys.includes(key)
                    const { diff, total } = differingPoints(response.predicted_arcs, run.response.predicted_arcs)
                    return (
                      <label key={key} className="flex cursor-pointer items-start gap-2 rounded px-1 py-1 hover:bg-raised/50">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) => toggleCompare(key, e.target.checked)}
                          className="mt-0.5 size-3.5 accent-[var(--ps-route)]"
                        />
                        <span className="min-w-0 flex-1">
                          <span className="flex items-center gap-1.5">
                            <svg width="18" height="8" aria-hidden="true" className="shrink-0">
                              <line
                                x1="1"
                                y1="4"
                                x2="17"
                                y2="4"
                                stroke={colorFor(key)}
                                strokeWidth="3"
                                strokeDasharray="4 3"
                                strokeLinecap="round"
                              />
                            </svg>
                            <span className="truncate text-[12.5px] text-ink">{runLabelOf(run)}</span>
                          </span>
                          <span className="mt-0.5 block text-[11px] leading-tight text-ink-faint">
                            Top-1 {fmtPct(run.response.metrics.top1_accuracy)} · Seq F1 {fmtPct(run.response.metrics.sequence_f1)} ·
                            differs on {fmtInt(diff)} of {fmtInt(total)}
                          </span>
                        </span>
                      </label>
                    )
                  })}
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-2 w-full"
                  onClick={() => void runQuickComparison()}
                  disabled={health.status !== 'online' || batch.running || !!quick?.running}
                >
                  <Play /> {quick?.running ? 'Comparing…' : 'Quick comparison'}
                </Button>
                {quick && quick.tripId === currentTripId && (
                  <p className="mt-1.5 text-[11.5px] leading-snug text-ink-faint">
                    {quick.running ? `${quick.done + 1} of ${quick.total}: ${quick.label}…` : (quick.error ?? 'Comparison ready.')}
                  </p>
                )}
                <p className="mt-1 text-[11px] leading-snug text-ink-faint">
                  Stored runs are reused. Overlays are drawn below the current route.
                </p>
              </div>
            )}
          </div>
        )}
        </div>
      )}

      {/* Trip caption */}
      {response && (
        <div className="pointer-events-none absolute bottom-3 left-3 z-[500] rounded-md border border-hairline bg-surface/95 px-2.5 py-1 text-[12px] leading-tight backdrop-blur">
          <span className="text-ink-faint">Trip </span>
          <span className="num text-[15px] font-semibold text-ink">{response.trip_id}</span>
          <span className="ml-2 text-ink-faint">{fmtInt(response.n_points)} GPS points</span>
        </div>
      )}

      {/* Focused point card */}
      {showPointCard && view && focusIndex !== null && focusIndex < view.pointInfo(0).total && (
        <div className="absolute bottom-12 left-3 z-[500] w-[260px] max-w-[calc(100%-1.5rem)] rounded-lg border border-hairline bg-surface/95 p-3 shadow-[0_8px_24px_rgb(0_0_0/0.35)] backdrop-blur">
          <PointDetails
            dense
            info={view.pointInfo(focusIndex)}
            onStep={(d) => setFocusIndex(Math.min(Math.max(0, focusIndex + d), view.pointInfo(0).total - 1))}
            onClose={() => setFocusIndex(null)}
          />
        </div>
      )}

      {basemap !== 'off' && tileErrors > 4 && (
        <div className="absolute right-3 bottom-14 z-[500] max-w-[260px] rounded-md border border-hairline bg-surface/95 px-2.5 py-1.5 text-[12px] text-ink-muted">
          Basemap tiles are not loading (no internet?). Trip layers still draw from the API data.
        </div>
      )}

      {view && view.layers.refused.length > 0 && (
        <div className="absolute right-3 bottom-24 z-[500] max-w-[280px] rounded-md border border-warn/40 bg-surface/95 px-2.5 py-1.5 text-[12px] text-warn">
          Not drawn: {view.layers.refused.join(', ')}. The UI only draws named layers and never displays reference-route data.
        </div>
      )}

      {overlay && <div className="absolute inset-0 z-[600] grid place-items-center bg-base/35 p-4">{overlay}</div>}
    </div>
  )
}

/** Predicted route: dark casing under a bright line, the way road maps draw highways. */
function RouteLine({ layer, placeholder }: { layer: MapLayer; placeholder: boolean }) {
  const data = layer.data as GeoJSON.FeatureCollection
  if (placeholder) {
    return (
      <GeoJSON
        data={data}
        interactive={false}
        pane={PANES.route}
        style={() => ({ color: MAP_COLORS.placeholder, weight: 2, opacity: 0.85, dashArray: '6 6' })}
      />
    )
  }
  return (
    <>
      <GeoJSON
        data={data}
        interactive={false}
        pane={PANES.route}
        style={() => ({ color: MAP_COLORS.routeCasing, weight: 7.5, opacity: 0.75, lineCap: 'round', lineJoin: 'round' })}
      />
      <GeoJSON
        data={data}
        pane={PANES.route}
        style={() => ({ color: MAP_COLORS.route, weight: 4, opacity: 1, lineCap: 'round', lineJoin: 'round' })}
        onEachFeature={(_f, l) => l.bindTooltip('Predicted route', { sticky: true, className: 'ps-tooltip' })}
      />
    </>
  )
}
