import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// host: true binds to 0.0.0.0 so the UI on Laptop 2 is reachable from the room (projector / evaluator).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  server: { host: true, port: 5173 },
  preview: { host: true, port: 4173 },
})
