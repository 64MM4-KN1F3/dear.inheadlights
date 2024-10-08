import { defineCollection, z } from 'astro:content';

const b = defineCollection({
    type: 'content',
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