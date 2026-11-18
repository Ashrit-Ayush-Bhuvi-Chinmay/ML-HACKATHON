import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/**
 * The single surface primitive. Panels carry a title row only when the content needs one;
 * `flush` removes inner padding for maps/tables that run edge to edge.
 */
export function Panel({
  title,
  description,
  actions,
  children,
  className,
  bodyClassName,
  flush = false,
  as: Tag = 'section',
}: {
  title?: ReactNode
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
  flush?: boolean
  as?: 'section' | 'div' | 'aside'
}) {
  return (
    <Tag className={cn('flex min-w-0 flex-col rounded-[10px] border border-hairline bg-surface', className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 px-4 pt-3.5 pb-3">
          <div className="min-w-0">
            {title && <h2 className="font-heading text-[18px] leading-tight font-semibold tracking-[0.01em] text-ink">{title}</h2>}
            {description && <p className="mt-0.5 text-[13px] leading-snug text-ink-muted">{description}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={cn(flush ? '' : 'px-4 pb-4', !title && !actions && !flush && 'pt-4', 'min-h-0 flex-1', bodyClassName)}>
        {children}
      </div>
    </Tag>
  )
}
