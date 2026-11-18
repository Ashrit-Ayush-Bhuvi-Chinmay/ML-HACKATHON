/**
 * Derived, read-only views of one /predict response, computed once per response object and
 * shared by the map, ribbon and inspectors. Nothing here is an evaluation metric.
 */
import type { PredictResponse } from '@/api/types'
import { boundsOf, extractLayers, isPlaceholderPrediction, type ExtractedLayers } from './geo'
import type { PointInfo } from '@/components/ribbon/PointDetails'
import { routePositions, runAt, runsOf, statsOf, type ArcRun, type SequenceStats } from './sequence'

export interface PredictionView {
  layers: ExtractedLayers
  bounds: [[number, number], [number, number]] | null
  runs: ArcRun[]
  stats: SequenceStats
  routePos: Map<number, number[]>
  placeholder: boolean
  pointInfo: (index: number) => PointInfo
}

const cache = new WeakMap<PredictResponse, PredictionView>()

export function viewOf(response: PredictResponse): PredictionView {
  const hit = cache.get(response)
  if (hit) return hit
  const layers = extractLayers(response.geojson)
  const runs = runsOf(response.predicted_arcs)
  const routePos = routePositions(response.route_arcs)
  const total = Math.max(response.predicted_arcs.length, layers.gpsPoints.length)
  const view: PredictionView = {
    layers,
    bounds: boundsOf(layers.layers),
    runs,
    stats: statsOf(response.predicted_arcs, runs),
    routePos,
    placeholder: isPlaceholderPrediction(response),
    pointInfo: (index) => {
      const arc = response.predicted_arcs[index]
      return {
        index,
        total,
        point: layers.gpsPoints[index] ?? null,
        arc,
        run: runAt(runs, index),
        routePos: arc === null || arc === undefined ? undefined : routePos.get(arc),
        routeLength: response.route_arcs.length,
      }
    },
  }
  cache.set(response, view)
  return view
}
