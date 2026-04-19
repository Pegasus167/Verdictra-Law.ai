import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ask': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/pdf': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})