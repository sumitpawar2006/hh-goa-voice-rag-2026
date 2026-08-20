import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy: Object.fromEntries(['/health','/query','/retrieve','/transcribe','/voice-query','/benchmark','/feedback'].map((path) => [path, 'http://127.0.0.1:8000'])) },
})
