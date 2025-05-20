import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../web_static"
  },
  server: {
    proxy: {
      '/predict': 'https://deepbooru.boon4681.com'
    }
  },
})
