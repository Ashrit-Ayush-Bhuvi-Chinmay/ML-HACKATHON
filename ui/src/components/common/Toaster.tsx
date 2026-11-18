import { useEffect, useState } from 'react'
import { CheckCircle2, CircleX, Info, X } from 'lucide-react'
import { dismissToast, subscribeToasts, type ToastItem } from '@/lib/toast'
import { cn } from '@/lib/utils'

const ICON = { success: CheckCircle2, error: CircleX, info: Info }
const COLOR = { success: 'text-ok', error: 'text-err', info: 'text-route' }

export function Toaster() {
  const [items, setItems] = useState<ToastItem[]>([])
  useEffect(() => subscribeToasts(setItems), [])

  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed top-[64px] left-1/2 z-[1600] flex w-[min(380px,calc(100vw-2rem))] -translate-x-1/2 flex-col gap-2"
    >
      {items.map((t) => {
        const Icon = ICON[t.tone]
        return (
          <div
            key={t.id}
            role={t.tone === 'error' ? 'alert' : 'status'}
            className="pointer-events-auto flex gap-2.5 rounded-lg border border-hairline bg-raised px-3.5 py-3 shadow-[0_12px_32px_rgb(0_0_0/0.4)] duration-200 animate-in fade-in slide-in-from-top-2"
          >
            <Icon className={cn('mt-0.5 size-4 shrink-0', COLOR[t.tone])} aria-hidden="true" />
            <div className="min-w-0 flex-1 text-[13px] leading-snug">
              <p className="font-medium text-ink">{t.title}</p>
              {t.detail && <p className="mt-0.5 text-ink-muted">{t.detail}</p>}
            </div>
            <button
              type="button"
              onClick={() => dismissToast(t.id)}
              className="-mt-0.5 -mr-1 grid size-6 place-items-center rounded text-ink-faint hover:bg-surface hover:text-ink"
              aria-label="Dismiss"
            >
              <X className="size-3.5" />
            </button>
          </div>
        )
      })}
    </div>
  )
}
