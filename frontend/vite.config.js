import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendPort = env.BACKEND_PORT || env.API_PORT || '8123'
  const apiUrl = env.VITE_API_URL || `http://localhost:${backendPort}`

  return {
    plugins: [vue()],
    server: {
      port: Number(env.VITE_PORT || 5173),
      host: env.VITE_HOST || 'localhost',
      proxy: {
        '/api': {
          target: apiUrl,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '')
        }
      }
    },
    build: {
      outDir: 'dist',
      sourcemap: false
    }
  }
})
