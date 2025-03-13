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
  build: {
    // Optimize build
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,  // Remove console.log in production
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          // Split vendor code for better caching
          vendor: ['vue', 'pinia', '@vueuse/core']
        }
      }
    }
  },
  optimizeDeps: {
    exclude: ['debug', 'browser-info', 'block-css-value', 'block-types', 'icon-badge-periods']
  }
}) 