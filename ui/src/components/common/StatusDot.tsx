import { cn } from '@/lib/utils'

export type DotTone = 'ok' | 'warn' | 'err' | 'idle' | 'busy'

const COLORS: Record<DotTone, string> = {
  ok: 'bg-ok',
  warn: 'bg-warn',
  err: 'bg-err',
  idle: 'bg-ink-faint',
  busy: 'bg-route',
}

/** Small status light. Always pair with a text label. */
export function StatusDot({ tone, pulse = false, className }: { tone: DotTone; pulse?: boolean; className?: string }) {
  return (
    <span className={cn('relative inline-flex size-2 shrink-0', className)} aria-hidden="true">
      {pulse && <span className={cn('absolute inset-0 animate-ping rounded-full opacity-60', COLORS[tone])} />}
      <span className={cn('relative inline-flex size-2 rounded-full', COLORS[tone])} />
    </span>
  )
}
