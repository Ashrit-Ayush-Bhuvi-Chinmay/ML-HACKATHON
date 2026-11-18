import { ChartColumn, ClipboardCheck, Compass, Map as MapIcon, Route, type LucideIcon } from 'lucide-react'

export interface NavItem {
  to: string
  label: string
  icon: LucideIcon
  /** Shown in the top bar under the page title. */
  summary: string
}

export const NAV: NavItem[] = [
  { to: '/', label: 'Overview', icon: Compass, summary: 'System status and how PathSnap works' },
  { to: '/trips', label: 'Trip explorer', icon: MapIcon, summary: 'Choose a trip, run the prediction, inspect the route' },
  { to: '/route', label: 'Route analysis', icon: Route, summary: 'Point-by-point arc matching for the selected trip' },
  { to: '/analytics', label: 'Model analytics', icon: ChartColumn, summary: 'Metrics and stage timings returned by the API' },
  { to: '/evaluation', label: 'Evaluation', icon: ClipboardCheck, summary: 'Run every trip and review the results together' },
]
