/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
  readonly VITE_PREDICT_TIMEOUT_MS?: string
  readonly VITE_TILE_URL?: string
  readonly VITE_TILE_ATTRIBUTION?: string
  readonly VITE_TILE_DARKEN?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
