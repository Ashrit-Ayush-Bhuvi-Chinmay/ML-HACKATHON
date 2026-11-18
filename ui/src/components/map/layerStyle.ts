import type { LayerKind } from '@/lib/geo'

/** Map layer colours, kept in sync with the CSS tokens in index.css. */
export const MAP_COLORS = {
  gps: '#5c9dff',
  route: '#38d3df',
  routeCasing: '#0e151d',
  road: '#7b889a',
  extra: '#b9a4f7',
  placeholder: '#b0bdcc',
  focus: '#ffffff',
  unmatched: '#e9eef5',
} as const

export const KIND_COLOR: Record<LayerKind, string> = {
  gps: MAP_COLORS.gps,
  route: MAP_COLORS.route,
  road: MAP_COLORS.road,
  extra: MAP_COLORS.extra,
}
