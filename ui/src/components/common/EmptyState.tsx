import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/** An empty area is an invitation to act: say what is missing and what to do. */
export function EmptyState({
  icon: Icon,
  title,
  children,
  action,
  className,
}: {
  icon: LucideIcon
  title: string
  children?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex flex-col items-center justify-center px-6 py-10 text-center', className)}>
      <div className="mb-3 grid size-10 place-items-center rounded-full border border-hairline bg-raised/60">
        <Icon className="size-[18px] text-ink-muted" aria-hidden="true" />
      </div>
      <p className="font-heading text-[17px] font-semibold text-ink">{title}</p>
      {children && <div className="mt-1 max-w-[46ch] text-[13px] leading-relaxed text-ink-muted">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
