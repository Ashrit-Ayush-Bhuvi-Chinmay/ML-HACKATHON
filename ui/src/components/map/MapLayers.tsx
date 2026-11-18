/**
 * Imperative Leaflet layers that react-leaflet's declarative components handle poorly:
 * thousands of indexed GPS points, highlights, the focus marker, controls and camera fitting.
 */
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import L from 'leaflet'
import { useMap } from 'react-leaflet'
import { MAP_COLORS } from './layerStyle'

/**
 * Draw order, bottom to top: road network, extra layers, GPS points, predicted route, highlight.
 * The route sits above the points so GPS drift away from the matched road stays visible.
 */
export const PANES = {
  road: 'ps-road',
  extra: 'ps-extra',
  gps: 'ps-gps',
  route: 'ps-route',
  highlight: 'ps-highlight',
} as const
const PANE_Z: Record<string, number> = { 'ps-road': 390, 'ps-extra': 395, 'ps-gps': 410, 'ps-route': 420, 'ps-highlight': 430 }

export function MapPanes() {
  const map = useMap()
  useLayoutEffect(() => {
    for (const [name, z] of Object.entries(PANE_Z)) {
      if (!map.getPane(name)) map.createPane(name).style.zIndex = String(z)
    }
  }, [map])
  return null
}

/** GeoJSON [lon, lat] -> Leaflet LatLng, via Leaflet's own converter. */
export const toLatLng = (p: [number, number]) => L.GeoJSON.coordsToLatLng(p)

/**
 * GPS observations. Matched points are filled blue dots, unmatched points (predicted arc null)
 * are hollow light rings, so the two read apart without relying on colour alone.
 * One shared tooltip follows the pointer instead of binding thousands of tooltips.
 */
export function GpsPointsLayer({
  points,
  arcs,
  showMatched,
  showUnmatched,
  uniform = false,
  onPick,
}: {
  points: [number, number][]
  arcs: (number | null)[]
  showMatched: boolean
  showUnmatched: boolean
  /** Draw every point as a plain GPS dot (placeholder output, where nothing was matched). */
  uniform?: boolean
  onPick: (index: number) => void
}) {
  const map = useMap()
  const renderer = useMemo(() => L.canvas({ padding: 0.3, tolerance: 5, pane: PANES.gps }), [])
  const pickRef = useRef(onPick)
  useEffect(() => {
    pickRef.current = onPick
  }, [onPick])

  useEffect(() => {
    if (points.length === 0 || (!showMatched && !showUnmatched)) return
    const group = L.layerGroup()
    const tip = L.tooltip({ className: 'ps-tooltip', direction: 'top', offset: [0, -6], opacity: 1 })
    const dense = points.length > 3000
    points.forEach((p, i) => {
      const arc = arcs[i] ?? null
      const matched = arc !== null
      if (!uniform && ((matched && !showMatched) || (!matched && !showUnmatched))) return
      const ll = toLatLng(p)
      const m = matched || uniform
        ? L.circleMarker(ll, {
            renderer,
            pane: PANES.gps,
            radius: dense ? 2.4 : 3,
            stroke: false,
            fillColor: MAP_COLORS.gps,
            fillOpacity: 0.92,
            bubblingMouseEvents: false,
          })
        : L.circleMarker(ll, {
            renderer,
            pane: PANES.gps,
            radius: dense ? 2.6 : 3.2,
            color: MAP_COLORS.unmatched,
            weight: 1.2,
            opacity: 0.85,
            fill: true,
            fillColor: MAP_COLORS.unmatched,
            fillOpacity: 0.05,
            bubblingMouseEvents: false,
          })
      m.on('click', () => pickRef.current(i))
      m.on('mouseover', () => {
        tip
          .setLatLng(ll)
          .setContent(`<b>Point ${i + 1}</b> &nbsp;${matched ? `arc ${arc}` : uniform ? 'no arc (placeholder)' : 'unmatched'}`)
          .addTo(map)
      })
      m.on('mouseout', () => tip.remove())
      group.addLayer(m)
    })
    group.addTo(map)
    return () => {
      tip.remove()
      group.remove()
    }
  }, [map, points, arcs, showMatched, showUnmatched, uniform, renderer])

  return null
}

/** Rings around a range of GPS points (the selected run), drawn above everything else. */
export function HighlightLayer({ points, range }: { points: [number, number][]; range: { start: number; end: number } | null }) {
  const map = useMap()
  const renderer = useMemo(() => L.canvas({ padding: 0.3, pane: PANES.highlight }), [])
  useEffect(() => {
    if (!range || points.length === 0) return
    const group = L.layerGroup()
    const end = Math.min(range.end, points.length - 1)
    for (let i = range.start; i <= end; i++) {
      group.addLayer(
        L.circleMarker(toLatLng(points[i]), {
          renderer,
          pane: PANES.highlight,
          radius: 4.5,
          color: MAP_COLORS.focus,
          weight: 1.5,
          fillColor: MAP_COLORS.gps,
          fillOpacity: 1,
          interactive: false,
        }),
      )
    }
    group.addTo(map)
    return () => {
      group.remove()
    }
  }, [map, points, range, renderer])
  return null
}

export function FocusMarker({ point }: { point: [number, number] | null }) {
  const map = useMap()
  useEffect(() => {
    if (!point) return
    const ll = toLatLng(point)
    const halo = L.circleMarker(ll, {
      radius: 12,
      color: MAP_COLORS.route,
      weight: 2.5,
      fill: false,
      opacity: 1,
      interactive: false,
      pane: 'markerPane',
    })
    const dot = L.circleMarker(ll, {
      radius: 5.5,
      color: MAP_COLORS.focus,
      weight: 2,
      fillColor: MAP_COLORS.gps,
      fillOpacity: 1,
      interactive: false,
      pane: 'markerPane',
    })
    halo.addTo(map)
    dot.addTo(map)
    return () => {
      halo.remove()
      dot.remove()
    }
  }, [map, point])
  return null
}

/** "Start" and "End" badges at the first and last GPS observation of the trip. */
export function TripEndpoints({ points }: { points: [number, number][] }) {
  const map = useMap()
  useEffect(() => {
    if (points.length < 2) return
    const badge = (text: string) =>
      L.divIcon({
        className: '',
        html: `<span class="ps-endpoint">${text}</span>`,
        iconSize: [0, 0],
      })
    const a = L.marker(toLatLng(points[0]), { icon: badge('Start'), interactive: false, keyboard: false })
    const b = L.marker(toLatLng(points[points.length - 1]), { icon: badge('End'), interactive: false, keyboard: false })
    a.addTo(map)
    b.addTo(map)
    return () => {
      a.remove()
      b.remove()
    }
  }, [map, points])
  return null
}

/**
 * Fits the camera when `fitKey` changes (new trip / new result / "Fit trip" button).
 * Leaves room on the right for the layers panel when the map is wide enough to show it beside the trip.
 */
export function FitBounds({
  bounds,
  fitKey,
  reserveRight,
}: {
  bounds: [[number, number], [number, number]] | null
  fitKey: string
  reserveRight: boolean
}) {
  const map = useMap()
  useEffect(() => {
    if (!bounds) return
    const wide = map.getSize().x > 760
    map.fitBounds(bounds, {
      paddingTopLeft: [60, 44],
      paddingBottomRight: [reserveRight && wide ? 268 : 44, 64],
      maxZoom: 17,
      animate: false,
    })
    // Only refit on an explicit key change, not on every bounds object identity change.
  }, [map, fitKey]) // eslint-disable-line react-hooks/exhaustive-deps
  return null
}

/** Pans to the focused point when it is outside the current view (e.g. after a ribbon click). */
export function PanToPoint({ point }: { point: [number, number] | null }) {
  const map = useMap()
  useEffect(() => {
    if (!point) return
    const ll = toLatLng(point)
    if (!map.getBounds().pad(-0.15).contains(ll)) map.panTo(ll, { animate: true, duration: 0.35 })
  }, [map, point])
  return null
}

/** Keeps Leaflet's size in sync with the flex layout and fullscreen changes. */
export function AutoResize() {
  const map = useMap()
  useEffect(() => {
    const el = map.getContainer()
    const ro = new ResizeObserver(() => map.invalidateSize({ animate: false }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [map])
  return null
}

/** Metric scale bar and a live cursor coordinate readout, updated without React re-renders. */
export function ScaleAndCoords() {
  const map = useMap()
  useEffect(() => {
    const scale = L.control.scale({ position: 'bottomright', imperial: false, maxWidth: 110 })
    const coords = new L.Control({ position: 'bottomright' })
    const el = L.DomUtil.create('div', 'ps-coords')
    el.textContent = 'lat, lon'
    coords.onAdd = () => el
    const onMove = (e: L.LeafletMouseEvent) => {
      el.textContent = `${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}`
    }
    coords.addTo(map)
    scale.addTo(map)
    map.on('mousemove', onMove)
    return () => {
      map.off('mousemove', onMove)
      coords.remove()
      scale.remove()
    }
  }, [map])
  return null
}
