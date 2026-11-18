import type { ReactNode } from 'react'
import { Info } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

/** Small "i" button that explains a term. Keyboard focusable; the tooltip opens on focus too. */
export function InfoTip({ children, label = 'More information', className }: { children: ReactNode; label?: string; className?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          className={cn(
            'inline-grid size-4 shrink-0 place-items-center rounded-full text-ink-faint transition-colors hover:text-ink focus-visible:text-ink',
            className,
          )}
        >
          <Info className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-[280px]">
        {children}
      </TooltipContent>
    </Tooltip>
  )
}
