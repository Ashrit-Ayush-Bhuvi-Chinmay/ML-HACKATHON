import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

export interface SegmentOption<T extends string> {
  value: T
  label: ReactNode
  /** Optional count shown after the label. */
  count?: number
}

/** Compact single-choice control (filters, view switches). Renders as a radio group. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  size = 'sm',
  className,
}: {
  value: T
  onChange: (v: T) => void
  options: SegmentOption<T>[]
  label: string
  size?: 'xs' | 'sm'
  className?: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className={cn('inline-flex rounded-md border border-hairline bg-inset p-0.5', className)}>
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-[5px] whitespace-nowrap transition-colors',
              size === 'xs' ? 'h-6 px-2 text-[12px]' : 'h-7 px-2.5 text-[12.5px]',
              active ? 'bg-raised text-ink shadow-[0_1px_0_rgb(255_255_255/0.04)_inset]' : 'text-ink-muted hover:text-ink',
            )}
          >
            {o.label}
            {o.count !== undefined && (
              <span className={cn('num text-[12px]', active ? 'text-ink-muted' : 'text-ink-faint')}>{o.count}</span>
            )}
          </button>
        )
      })}
    </div>
  )
}
