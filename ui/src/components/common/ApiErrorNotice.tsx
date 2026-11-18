import { RotateCw } from 'lucide-react'
import type { ApiError } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Notice } from './Notice'

const TITLES: Record<ApiError['kind'], string> = {
  network: 'API unreachable',
  timeout: 'API timed out',
  not_found: 'Not found on the API',
  server: 'API error',
  client: 'Request rejected',
  invalid_response: 'Unexpected API response',
  cancelled: 'Request cancelled',
}

export function ApiErrorNotice({ error, onRetry, className }: { error: ApiError; onRetry?: () => void; className?: string }) {
  const tone = error.kind === 'cancelled' ? 'info' : 'error'
  return (
    <Notice
      tone={tone}
      className={className}
      title={
        <>
          {TITLES[error.kind]}
          {error.status ? <span className="ml-1.5 font-normal text-ink-muted">HTTP {error.status}</span> : null}
        </>
      }
      action={
        onRetry && (
          <Button size="sm" variant="outline" onClick={onRetry}>
            <RotateCw /> Try again
          </Button>
        )
      }
    >
      {error.message}
    </Notice>
  )
}
