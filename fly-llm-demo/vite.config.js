import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  base: './',
  plugins: [svelte()],
  optimizeDeps: { exclude: ['onnxruntime-web', '@huggingface/transformers'] },
  build: { target: 'esnext', chunkSizeWarningLimit: 4000 },
});
