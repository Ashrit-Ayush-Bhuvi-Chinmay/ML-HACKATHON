/** Minimal toast bus. UI-free so state code can announce outcomes. */
export type ToastTone = 'success' | 'error' | 'info'

export interface ToastItem {
  id: number
  tone: ToastTone
  title: string
  detail?: string
}

type Listener = (items: ToastItem[]) => void

let items: ToastItem[] = []
let nextId = 1
const listeners = new Set<Listener>()
const emit = () => listeners.forEach((l) => l(items))

export function toast(tone: ToastTone, title: string, detail?: string, ttlMs = 3500) {
  const item = { id: nextId++, tone, title, detail }
  items = [...items, item].slice(-4)
  emit()
  window.setTimeout(() => dismissToast(item.id), ttlMs)
}

export function dismissToast(id: number) {
  items = items.filter((t) => t.id !== id)
  emit()
}

export function subscribeToasts(l: Listener): () => void {
  listeners.add(l)
  l(items)
  return () => listeners.delete(l)
}
