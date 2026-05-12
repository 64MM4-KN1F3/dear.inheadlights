import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const b = defineCollection({
    loader: glob({ pattern: '**/[^_]*.{md,mdx}', base: "./src/content/b" }),
    schema: z.object({
        title: z.string(),
        publishDate: z.date(),
        draft: z.boolean().optional().default(false),
        tags: z.array(z.string()),
    }),
});

export const collections = {
    b
};
