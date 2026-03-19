import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteSourceLocator } from '@metagptx/vite-plugin-source-locator'


// https://vitejs.dev/config/
export default defineConfig({
  plugins: [viteSourceLocator({
    prefix: 'mgx'
  }), react()],
})
