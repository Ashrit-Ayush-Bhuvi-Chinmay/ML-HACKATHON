/**
 * Splits the /predict GeoJSON into map layers by properties.layer.
 * Coordinates stay in GeoJSON [lon, lat] order; Leaflet's L.geoJSON handles the swap.
 */
import type { Feature, FeatureCollection, PredictResponse, Position } from '@/api/types'

export type LayerKind = 'gps' | 'route' | 'road' | 'extra'

export interface MapLayer {
  key: string
  label: string
  kind: LayerKind
  data: FeatureCollection
  vertexCount: number
}

export interface ExtractedLayers {
  layers: MapLayer[]
  /** GPS observations in order, [lon, lat]. Index i corresponds to predicted_arcs[i]. */
  gpsPoints: [number, number][]
  /** Layers refused because their name suggests reference-route / ground-truth data. */
  refused: string[]
}

const KNOWN: Record<string, { label: string; kind: LayerKind; order: number }> = {
  road_network: { label: 'Road network', kind: 'road', order: 0 },
  roads: { label: 'Road network', kind: 'road', order: 0 },
  predicted_route: { label: 'Predicted route', kind: 'route', order: 2 },
  gps: { label: 'GPS observations', kind: 'gps', order: 3 },
}

/**
 * The reference route must never reach the UI (evaluation integrity rule). This is defence in
 * depth: the real guarantee is that the API never sends it. A layer whose name looks like
 * reference / ground-truth data is not drawn, and neither is a feature without a layer name.
 * "gt" is matched as a separate token so names such as gt_route or route_gt are caught.
 */
const REFERENCE_PATTERN = /reference|truth|(^|[^a-z])gt([^a-z]|$)|true[\s_-]?route|actual[\s_-]?route|\.route\b/i
export const UNLABELLED = 'features without a layer name'

export function isRefusedLayerName(name: string): boolean {
  return REFERENCE_PATTERN.test(name)
}

function countVertices(f: Feature): number {
  const g = f.geometry
  if (!g) return 0
  switch (g.type) {
    case 'Point':
      return 1
    case 'MultiPoint':
    case 'LineString':
      return g.coordinates.length
    case 'MultiLineString':
    case 'Polygon':
      return g.coordinates.reduce((n, l) => n + l.length, 0)
    case 'MultiPolygon':
      return g.coordinates.reduce((n, p) => n + p.reduce((m, l) => m + l.length, 0), 0)
    default:
      return 0
  }
}

function humanise(key: string): string {
  const s = key.replace(/[_-]+/g, ' ').trim()
  return s ? s[0].toUpperCase() + s.slice(1) : 'Unnamed layer'
}

const toLonLat = (p: Position): [number, number] => [p[0], p[1]]

export function extractLayers(fc: FeatureCollection): ExtractedLayers {
  const groups = new Map<string, Feature[]>()
  const refused = new Set<string>()
  for (const f of fc.features) {
    const name = typeof f.properties?.layer === 'string' ? f.properties.layer.trim() : ''
    if (!name) {
      refused.add(UNLABELLED)
      continue
    }
    if (isRefusedLayerName(name)) {
      refused.add(name)
      continue
    }
    const key = name
    const list = groups.get(key) ?? []
    list.push(f)
    groups.set(key, list)
  }

  const layers: MapLayer[] = [...groups.entries()]
    .map(([key, features]) => {
      const known = KNOWN[key]
      return {
        key,
        label: known?.label ?? humanise(key),
        kind: known?.kind ?? 'extra',
        data: { type: 'FeatureCollection' as const, features },
        vertexCount: features.reduce((n, f) => n + countVertices(f), 0),
        order: known?.order ?? 1,
      }
    })
    .sort((a, b) => a.order - b.order)
    .map(({ order: _order, ...layer }) => layer)

  const gpsPoints: [number, number][] = []
  for (const f of groups.get('gps') ?? []) {
    const g = f.geometry
    if (g?.type === 'MultiPoint') for (const p of g.coordinates) gpsPoints.push(toLonLat(p))
    else if (g?.type === 'Point') gpsPoints.push(toLonLat(g.coordinates))
  }

  return { layers, gpsPoints, refused: [...refused] }
}

/** [south-west, north-east] in Leaflet [lat, lon] order, or null if there is nothing to frame. */
export function boundsOf(layers: MapLayer[]): [[number, number], [number, number]] | null {
  let minLat = Infinity
  let minLon = Infinity
  let maxLat = -Infinity
  let maxLon = -Infinity
  const visit = (p: Position) => {
    const [lon, lat] = p
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return
    if (lat < minLat) minLat = lat
    if (lat > maxLat) maxLat = lat
    if (lon < minLon) minLon = lon
    if (lon > maxLon) maxLon = lon
  }
  // Frame the trip (GPS + route), not the whole road network, which can be much larger.
  const framing = layers.filter((l) => l.kind === 'gps' || l.kind === 'route')
  for (const layer of framing.length ? framing : layers) {
    for (const f of layer.data.features) {
      const g = f.geometry
      if (!g) continue
      if (g.type === 'Point') visit(g.coordinates)
      else if (g.type === 'MultiPoint' || g.type === 'LineString') g.coordinates.forEach(visit)
      else if (g.type === 'MultiLineString' || g.type === 'Polygon') g.coordinates.forEach((l) => l.forEach(visit))
      else if (g.type === 'MultiPolygon') g.coordinates.forEach((p) => p.forEach((l) => l.forEach(visit)))
    }
  }
  if (!Number.isFinite(minLat)) return null
  return [
    [minLat, minLon],
    [maxLat, maxLon],
  ]
}

/**
 * True when the response is the contract's placeholder output: no arc matched and
 * no route, so the "predicted_route" layer is just the raw GPS polyline.
 */
export function isPlaceholderPrediction(p: Pick<PredictResponse, 'predicted_arcs' | 'route_arcs'>): boolean {
  return p.route_arcs.length === 0 && p.predicted_arcs.every((a) => a === null)
}
