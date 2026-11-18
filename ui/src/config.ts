/**
 * Runtime configuration.
 *
 * API base URL precedence (highest first):
 *   1. Address saved in the UI ("API address" in the top bar), stored in localStorage.
 *      Lets you point Laptop 2 at Laptop 1's current IP on demo day without rebuilding.
 *   2. VITE_API_BASE_URL from ui/.env.local (read at build / dev-server start).
 *   3. http://localhost:8000 (only correct when API and UI run on the same machine).
 */

const STORAGE_KEY = 'pathsnap.apiBaseUrl'
// 127.0.0.1, not localhost: on Windows "localhost" may resolve to IPv6 (::1) first while the API
// listens on IPv4 only, which made requests time out.
const FALLBACK_URL = 'http://127.0.0.1:8000'

export type ApiUrlSource = 'saved' | 'env' | 'default'

function normalise(url: string): string {
  return url.trim().replace(/\/+$/, '')
}

export function isValidHttpUrl(value: string): boolean {
  try {
    const u = new URL(value.trim())
    return u.protocol === 'http:' || u.protocol === 'https:'
  } catch {
    return false
  }
}

function readSaved(): string | null {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY)
    return v && isValidHttpUrl(v) ? normalise(v) : null
  } catch {
    return null
  }
}

const envUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim()

export function resolveApiBaseUrl(): { url: string; source: ApiUrlSource } {
  const saved = readSaved()
  if (saved) return { url: saved, source: 'saved' }
  if (envUrl && isValidHttpUrl(envUrl)) return { url: normalise(envUrl), source: 'env' }
  return { url: FALLBACK_URL, source: 'default' }
}

export function saveApiBaseUrl(url: string | null): void {
  try {
    if (url === null) window.localStorage.removeItem(STORAGE_KEY)
    else window.localStorage.setItem(STORAGE_KEY, normalise(url))
  } catch {
    /* storage unavailable: setting only lasts for this page load */
  }
}

export const envApiBaseUrl = envUrl && isValidHttpUrl(envUrl) ? normalise(envUrl) : null

/** Prediction can be slow (candidate generation + HMM over thousands of points). */
export const PREDICT_TIMEOUT_MS = Number(import.meta.env.VITE_PREDICT_TIMEOUT_MS) || 180_000
export const DEFAULT_TIMEOUT_MS = 10_000
export const HEALTH_POLL_MS = 15_000

/**
 * Basemap tiles. Trip layers still render if tiles cannot load (offline).
 * Default: OpenStreetMap standard tiles (no API key), darkened with a CSS filter so the
 * GPS and route layers stay the brightest things on the map. Set VITE_TILE_DARKEN=false
 * when VITE_TILE_URL already points at a dark style.
 */
const customTiles = (import.meta.env.VITE_TILE_URL as string | undefined)?.trim()
export const TILE_URL = customTiles || 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
export const TILE_ATTRIBUTION =
  (import.meta.env.VITE_TILE_ATTRIBUTION as string | undefined) ||
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
export const TILE_DARKEN = (import.meta.env.VITE_TILE_DARKEN as string | undefined) !== 'false'
