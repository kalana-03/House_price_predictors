// vite.config.ts   (or .js)
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'          // ← usually already here
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),          // ← add this line
  ],
})