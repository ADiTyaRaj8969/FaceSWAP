import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        timeout: 300000,       // 5 minutes socket timeout
        proxyTimeout: 300000,  // 5 minutes proxy timeout
      },
    },
  },
  build: {
    outDir: '../static/react',
    emptyOutDir: true,
  },
});
