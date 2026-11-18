import type { ReactNode } from 'react'
import { AlertTriangle, CircleAlert, FlaskConical, Info, ShieldCheck } from 'lucide-react'
import { cn } from '@/lib/utils'

type Tone = 'info' | 'warn' | 'error' | 'stub' | 'secure'

const TONES: Record<Tone, { icon: typeof Info; box: string; iconColor: string }> = {
  info: { icon: Info, box: 'border-hairline bg-raised/50', iconColor: 'text-ink-muted' },
  warn: { icon: AlertTriangle, box: 'border-warn/35 bg-warn/[0.07]', iconColor: 'text-warn' },
  error: { icon: CircleAlert, box: 'border-err/40 bg-err/[0.08]', iconColor: 'text-err' },
  stub: { icon: FlaskConical, box: 'border-warn/35 bg-warn/[0.07]', iconColor: 'text-warn' },
  secure: { icon: ShieldCheck, box: 'border-hairline bg-raised/40', iconColor: 'text-ok' },
}

/** Inline message with an icon. Status colour is never the only signal: icon + title carry it too. */
export function Notice({
  tone = 'info',
  title,
  children,
  action,
  className,
}: {
  tone?: Tone
  title: ReactNode
  children?: ReactNode
  action?: ReactNode
  className?: string
}) {
  const t = TONES[tone]
  const Icon = t.icon
  return (
    <div role={tone === 'error' ? 'alert' : 'status'} className={cn('flex gap-2.5 rounded-lg border px-3 py-2.5', t.box, className)}>
      <Icon className={cn('mt-0.5 size-4 shrink-0', t.iconColor)} aria-hidden="true" />
      <div className="min-w-0 flex-1 text-[13px] leading-snug">
        <p className="font-medium text-ink">{title}</p>
        {children && <div className="mt-0.5 text-ink-muted">{children}</div>}
        {action && <div className="mt-2">{action}</div>}
      </div>
    </div>
  )
}
