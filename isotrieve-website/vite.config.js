import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    open: true
  },
  publicDir: 'static',
  build: {
    rollupOptions: {
      input: resolve(__dirname, 'index.html')
    }
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    }
  }
})
