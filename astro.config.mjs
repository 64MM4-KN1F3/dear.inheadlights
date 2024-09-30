import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import { targetBlank } from './src/plugins/targetBlank';

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind()],
  site: 'https://inheadlights.com',
  markdown: {
    rehypePlugins: [[targetBlank, { domain: 'inheadlights.com' }]],
  },
});