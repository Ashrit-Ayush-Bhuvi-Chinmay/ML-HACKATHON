import { useState, type FormEvent } from 'react'
import { RotateCw, Server } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { StatusDot, type DotTone } from '@/components/common/StatusDot'
import { envApiBaseUrl, isValidHttpUrl } from '@/config'
import { fmtMs, fmtRelative } from '@/lib/format'
import { useSession } from '@/state/session'

function hostOf(url: string) {
  try {
    return new URL(url).host
  } catch {
    return url
  }
}

const SOURCE_LABEL = {
  saved: 'Saved in this browser',
  env: 'From VITE_API_BASE_URL',
  default: 'Default (same machine)',
} as const

/** Top-bar backend indicator. Opens a panel with health details and the API address setting. */
export function ApiStatus() {
  const { health, healthChecking, refreshHealth, apiUrl, apiUrlSource, setApiUrl, isStubModel } = useSession()
  const [draft, setDraft] = useState(apiUrl)
  const [open, setOpen] = useState(false)
  const draftValid = isValidHttpUrl(draft)

  const tone: DotTone = health.status === 'online' ? (isStubModel ? 'warn' : 'ok') : health.status === 'offline' ? 'err' : 'busy'
  const label = health.status === 'online' ? 'API online' : health.status === 'offline' ? 'API offline' : 'Connecting'

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!draftValid) return
    setApiUrl(draft)
    setOpen(false)
  }

  return (
    <Popover
      open={open}
      onOpenChange={(o) => {
        setOpen(o)
        if (o) setDraft(apiUrl)
      }}
    >
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex h-8 items-center gap-2 rounded-md border border-hairline bg-surface px-2.5 text-[13px] text-ink transition-colors hover:bg-raised"
          aria-label={`${label} at ${hostOf(apiUrl)}. Open API settings`}
        >
          <StatusDot tone={tone} pulse={health.status === 'checking'} />
          <span>{label}</span>
          <span className="hidden text-ink-faint lg:inline">{hostOf(apiUrl)}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[340px] border-hairline bg-raised p-0">
        <div className="space-y-3 p-4">
          <div className="flex items-center justify-between">
            <p className="font-heading text-[16px] font-semibold">Backend</p>
            <Button size="xs" variant="ghost" onClick={refreshHealth} disabled={healthChecking}>
              <RotateCw className={healthChecking ? 'animate-spin' : ''} /> Check now
            </Button>
          </div>

          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
            <dt className="text-ink-muted">Status</dt>
            <dd className="flex items-center gap-1.5">
              <StatusDot tone={tone} /> {health.status === 'online' ? health.data.status : label}
            </dd>
            <dt className="text-ink-muted">Model</dt>
            <dd>{health.data ? health.data.model : 'N/A'}</dd>
            <dt className="text-ink-muted">Trips</dt>
            <dd>{health.data ? health.data.n_trips : 'N/A'}</dd>
            <dt className="text-ink-muted">Latency</dt>
            <dd>{health.status === 'online' ? fmtMs(health.latencyMs) : 'N/A'}</dd>
            <dt className="text-ink-muted">Checked</dt>
            <dd>{health.checkedAt ? fmtRelative(health.checkedAt) : 'N/A'}</dd>
          </dl>

          {health.status === 'offline' && <p className="text-[12px] leading-snug text-err">{health.error.message}</p>}
        </div>

        <form onSubmit={submit} className="space-y-2 border-t border-hairline p-4">
          <label htmlFor="api-url" className="flex items-center gap-1.5 text-[13px] font-medium">
            <Server className="size-3.5 text-ink-muted" /> API address
          </label>
          <Input
            id="api-url"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="http://192.168.1.20:8000"
            spellCheck={false}
            aria-invalid={!draftValid}
            className="h-8 bg-surface text-[13px] text-ink"
          />
          <p className="text-[12px] text-ink-faint">
            {SOURCE_LABEL[apiUrlSource]}. Use Laptop 1&rsquo;s IP from <code className="text-ink-muted">ipconfig</code>, port 8000.
          </p>
          <div className="flex justify-end gap-1.5 pt-1">
            {apiUrlSource === 'saved' && (
              <Button type="button" size="sm" variant="ghost" onClick={() => (setApiUrl(null), setOpen(false))}>
                {envApiBaseUrl ? 'Use .env address' : 'Reset'}
              </Button>
            )}
            <Button type="submit" size="sm" disabled={!draftValid || draft.replace(/\/+$/, '') === apiUrl}>
              Connect
            </Button>
          </div>
        </form>
      </PopoverContent>
    </Popover>
  )
}

/** Model badge: separates a running stub from a trained model. */
export function ModelBadge() {
  const { health, isStubModel } = useSession()
  if (!health.data) return null
  return (
    <span
      className={
        'hidden h-8 items-center gap-1.5 rounded-md border px-2.5 text-[13px] sm:flex ' +
        (isStubModel ? 'border-warn/40 text-warn' : 'border-hairline text-ink')
      }
      title={isStubModel ? 'The API is running the placeholder pipeline: no trained model is loaded.' : 'Model reported by /health'}
    >
      <span className="text-ink-muted">Model</span>
      <span className="font-medium">{isStubModel ? 'stub (placeholder)' : health.data.model}</span>
    </span>
  )
}
