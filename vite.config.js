import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    hmr: {
      overlay: false // Disable error overlay
    }
  },
  optimizeDeps: {
    exclude: ['debug', 'browser-info', 'block-css-value', 'block-types', 'icon-badge-periods']
  }
}) 