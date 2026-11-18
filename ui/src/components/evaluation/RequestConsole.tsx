import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, Copy, Terminal, Trash2 } from 'lucide-react'
import { clearRequestLog, subscribeRequestLog, type RequestLogEntry } from '@/api/client'
import { Segmented } from '@/components/common/Segmented'
import { Button } from '@/components/ui/button'
import { fmtMs } from '@/lib/format'
import { cn } from '@/lib/utils'

function clock(ts: number) {
  const d = new Date(ts)
  return `${d.toLocaleTimeString('en-GB', { hour12: false })}.${String(d.getMilliseconds()).padStart(3, '0')}`
}

function line(e: RequestLogEntry) {
  const status = e.outcome === 'pending' ? '...' : (e.status ?? 'ERR')
  const dur = e.durationMs === null ? '' : fmtMs(e.durationMs)
  return `${clock(e.startedAt)}  ${e.method.padEnd(4)} ${e.path.padEnd(9)} ${String(status).padEnd(4)} ${dur.padStart(8)}  ${e.note ?? ''}${e.message ? `  ${e.message}` : ''}`
}

/** Browser-side request log styled as an operations console. Oldest first, follows new lines. */
export function RequestConsole() {
  const [entries, setEntries] = useState<RequestLogEntry[]>([])
  const [filter, setFilter] = useState<'all' | 'predict' | 'errors'>('all')
  const [copied, setCopied] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)
  const stick = useRef(true)

  useEffect(() => subscribeRequestLog(setEntries), [])

  const shown = useMemo(() => {
    const list = [...entries].reverse()
    if (filter === 'errors') return list.filter((e) => e.outcome === 'error')
    if (filter === 'predict') return list.filter((e) => e.path === '/predict')
    return list
  }, [entries, filter])

  useEffect(() => {
    const el = scroller.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [shown])

  const errors = entries.filter((e) => e.outcome === 'error').length
  const base = entries[0]?.baseUrl

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(shown.map(line).join('\n'))
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable (insecure context): ignore */
    }
  }

  return (
    <section className="overflow-hidden rounded-[10px] border border-hairline bg-surface" aria-label="Request log">
      <header className="flex flex-wrap items-center justify-between gap-2 px-4 pt-3.5 pb-3">
        <div>
          <h2 className="flex items-center gap-2 font-heading text-[18px] leading-tight font-semibold">
            <Terminal className="size-4 text-ink-muted" /> Request log
          </h2>
          <p className="mt-0.5 text-[13px] text-ink-muted">Calls made from this browser. Compare them with the API console on Laptop 1.</p>
        </div>
        <div className="flex items-center gap-1.5">
          <Segmented
            label="Filter log"
            size="xs"
            value={filter}
            onChange={setFilter}
            options={[
              { value: 'all', label: 'All', count: entries.length },
              { value: 'predict', label: '/predict' },
              { value: 'errors', label: 'Errors', count: errors },
            ]}
          />
          <Button size="sm" variant="ghost" onClick={() => void copy()} disabled={!shown.length}>
            {copied ? <Check /> : <Copy />} {copied ? 'Copied' : 'Copy'}
          </Button>
          <Button size="sm" variant="ghost" onClick={clearRequestLog} disabled={!entries.length}>
            <Trash2 /> Clear
          </Button>
        </div>
      </header>

      <div className="border-t border-hairline bg-inset">
        <div className="flex items-center justify-between gap-2 border-b border-hairline/70 px-4 py-1.5 text-[11.5px] text-ink-faint">
          <span className="mono truncate">{base ? `api ${base}` : 'api (no requests yet)'}</span>
          <span className="mono shrink-0">{shown.length} lines</span>
        </div>
        <div
          ref={scroller}
          onScroll={(e) => {
            const el = e.currentTarget
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
          }}
          className="scroll-thin mono h-[260px] overflow-y-auto px-4 py-2 text-[12.5px] leading-[1.7]"
          role="log"
          aria-live="off"
        >
          {shown.length === 0 ? (
            <p className="text-ink-faint">{entries.length ? 'No lines match this filter.' : 'Waiting for the first request.'}</p>
          ) : (
            shown.map((e) => (
              <div key={e.id} className={cn('flex flex-wrap gap-x-3', e.outcome === 'error' && 'bg-err/[0.06]')}>
                <span className="text-ink-faint">{clock(e.startedAt)}</span>
                <span className={cn('w-10', e.method === 'POST' ? 'text-route' : 'text-ink-muted')}>{e.method}</span>
                <span className="w-[72px] text-ink">{e.path}</span>
                <span
                  className={cn(
                    'w-9',
                    e.outcome === 'pending' ? 'text-route' : e.outcome === 'ok' ? 'text-ok' : 'text-err',
                  )}
                >
                  {e.outcome === 'pending' ? '...' : (e.status ?? 'ERR')}
                </span>
                <span className="w-[64px] text-right text-ink-muted">{e.durationMs === null ? '' : fmtMs(e.durationMs)}</span>
                {e.note && <span className="text-ink-muted">{e.note}</span>}
                {e.message && <span className="basis-full pl-[calc(12ch+0.75rem)] text-err/90">{e.message}</span>}
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  )
}
