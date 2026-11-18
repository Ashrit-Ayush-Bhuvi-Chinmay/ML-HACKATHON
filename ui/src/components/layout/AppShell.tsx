import { Outlet, useLocation } from 'react-router'
import { BrandMark } from '@/components/common/BrandMark'
import { Toaster } from '@/components/common/Toaster'
import { ApiStatus, ModelBadge } from './ApiStatus'
import { NAV } from './nav'
import { MobileNav, Sidebar } from './Sidebar'

export function AppShell() {
  const { pathname } = useLocation()
  const page = NAV.find((n) => (n.to === '/' ? pathname === '/' : pathname.startsWith(n.to))) ?? NAV[0]

  return (
    <div className="flex min-h-dvh bg-base">
      <a
        href="#main"
        className="sr-only z-[2000] rounded bg-raised px-3 py-2 focus:not-sr-only focus:fixed focus:top-2 focus:left-2"
      >
        Skip to content
      </a>
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-[1000] flex h-14 shrink-0 items-center gap-3 border-b border-hairline/80 bg-base/92 px-4 backdrop-blur md:px-6">
          <BrandMark className="size-6 md:hidden" />
          <div className="min-w-0 flex-1">
            <h1 className="truncate font-heading text-[21px] leading-tight font-semibold tracking-[0.01em]">{page.label}</h1>
            <p className="hidden truncate text-[12px] text-ink-faint sm:block">{page.summary}</p>
          </div>
          <ModelBadge />
          <ApiStatus />
        </header>
        <main id="main" key={page.to} className="flex min-h-0 flex-1 flex-col px-4 pt-4 pb-20 duration-200 animate-in fade-in md:px-6 md:pb-6">
          <Outlet />
        </main>
      </div>
      <MobileNav />
      <Toaster />
    </div>
  )
}
