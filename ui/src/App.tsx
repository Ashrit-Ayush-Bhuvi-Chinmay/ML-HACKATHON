import { lazy, Suspense, type ReactNode } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router'
import { TooltipProvider } from '@/components/ui/tooltip'
import { AppShell } from '@/components/layout/AppShell'
import { SessionProvider } from '@/state/session'

// Pages load on first visit to keep the app shell light; the map (Leaflet) and charts (Recharts) stay out of it.
const Overview = lazy(() => import('@/pages/Overview').then((m) => ({ default: m.Overview })))
const TripExplorer = lazy(() => import('@/pages/TripExplorer').then((m) => ({ default: m.TripExplorer })))
const RouteAnalysis = lazy(() => import('@/pages/RouteAnalysis').then((m) => ({ default: m.RouteAnalysis })))
const ModelAnalytics = lazy(() => import('@/pages/ModelAnalytics').then((m) => ({ default: m.ModelAnalytics })))
const Evaluation = lazy(() => import('@/pages/Evaluation').then((m) => ({ default: m.Evaluation })))

const page = (el: ReactNode) => (
  <Suspense fallback={<div className="p-6 text-[13px] text-ink-faint">Loading</div>}>{el}</Suspense>
)

export default function App() {
  return (
    <SessionProvider>
      <TooltipProvider delayDuration={200}>
        <BrowserRouter>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={page(<Overview />)} />
              <Route path="trips" element={page(<TripExplorer />)} />
              <Route path="route" element={page(<RouteAnalysis />)} />
              <Route path="analytics" element={page(<ModelAnalytics />)} />
              <Route path="evaluation" element={page(<Evaluation />)} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </SessionProvider>
  )
}
