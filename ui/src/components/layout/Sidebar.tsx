import { NavLink } from 'react-router'
import { BrandMark } from '@/components/common/BrandMark'
import { tripStatus } from '@/lib/tripStatus'
import { cn } from '@/lib/utils'
import { useSession } from '@/state/session'
import { NAV } from './nav'

const STATUS_TEXT = { 'not-run': 'Not run', running: 'Predicting', done: 'Predicted', failed: 'Failed' } as const
const STATUS_DOT = { 'not-run': 'bg-ink-faint', running: 'bg-route', done: 'bg-ok', failed: 'bg-err' } as const

/**
 * Navigation rail. Full width from 1600px; below that it collapses to icons so the
 * map workspace keeps its width at projector resolutions (1280x720, 1366x768).
 */
export function Sidebar() {
  const { selectedTripId, predictions } = useSession()
  const status = tripStatus(selectedTripId ? predictions[selectedTripId] : undefined)

  return (
    <aside className="sticky top-0 hidden h-dvh w-16 shrink-0 flex-col border-r border-hairline/80 bg-rail md:flex min-[1600px]:w-[224px]">
      <div className="flex h-14 items-center justify-center gap-2.5 px-3 min-[1600px]:justify-start min-[1600px]:px-4">
        <BrandMark />
        <div className="hidden leading-none min-[1600px]:block">
          <p className="font-heading text-[20px] font-semibold tracking-[0.01em] text-ink">PathSnap</p>
          <p className="mt-0.5 text-[11.5px] text-ink-faint">Map-matching workspace</p>
        </div>
      </div>

      <nav aria-label="Main" className="mt-3 flex flex-col gap-1 px-2">
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            title={label}
            aria-label={label}
            className={({ isActive }) =>
              cn(
                'group relative flex h-10 items-center justify-center gap-2.5 rounded-md text-[14px] transition-colors min-[1600px]:justify-start min-[1600px]:px-3',
                isActive ? 'bg-surface text-ink shadow-[0_0_0_1px_var(--ps-hairline)_inset]' : 'text-ink-muted hover:bg-surface/60 hover:text-ink',
              )
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={cn('absolute top-2 bottom-2 left-0 w-[3px] rounded-r-full', isActive ? 'bg-route' : 'bg-transparent')}
                  aria-hidden="true"
                />
                <Icon className={cn('size-[18px] shrink-0', isActive ? 'text-route' : 'text-ink-faint group-hover:text-ink-muted')} />
                <span className="hidden min-[1600px]:inline">{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto border-t border-hairline/80 px-2 py-3 min-[1600px]:px-4">
        {selectedTripId ? (
          <div title={`Trip ${selectedTripId}: ${STATUS_TEXT[status]}`}>
            <p className="hidden text-[12px] text-ink-faint min-[1600px]:block">Selected trip</p>
            <div className="mt-1 flex flex-col items-center gap-1 min-[1600px]:flex-row min-[1600px]:items-baseline min-[1600px]:justify-between min-[1600px]:gap-2">
              <span className="num text-[12px] font-semibold text-ink min-[1600px]:text-[21px]">{selectedTripId}</span>
              <span className="flex items-center gap-1.5 text-[12px] text-ink-muted">
                <span className={cn('size-2 rounded-full', STATUS_DOT[status])} aria-hidden="true" />
                <span className="hidden min-[1600px]:inline">{STATUS_TEXT[status]}</span>
              </span>
            </div>
          </div>
        ) : (
          <p className="hidden text-[13px] text-ink-faint min-[1600px]:block">No trip selected</p>
        )}
      </div>
    </aside>
  )
}

/** Bottom tab bar for narrow screens. */
export function MobileNav() {
  return (
    <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-[1100] flex border-t border-hairline bg-rail/95 backdrop-blur md:hidden">
      {NAV.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) => cn('flex flex-1 flex-col items-center gap-0.5 py-2 text-[10.5px]', isActive ? 'text-route' : 'text-ink-muted')}
        >
          <Icon className="size-[18px]" />
          <span className="truncate">{label.split(' ')[0]}</span>
        </NavLink>
      ))}
    </nav>
  )
}
