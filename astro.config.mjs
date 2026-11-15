import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import { targetBlank } from './src/plugins/targetBlank';

import expressiveCode from 'astro-expressive-code';

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind(), expressiveCode(), mdx()],
  site: 'https://inheadlights.com',
  output: 'static',
  markdown: {
    rehypePlugins: [[targetBlank, { domain: 'inheadlights.com' }]],
  },
});