/** @type {import('tailwindcss').Config} */
const plugin = require('tailwindcss/plugin');

export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: {
				'warm-light': '#f2f0e8',
			},
		},
	},
	darkMode: 'selector',
	plugins: [
		require('@tailwindcss/typography'),
		plugin(function ({ addUtilities }) {
			const newUtilities = {
				'.text-rainbow': {
					'@apply font-bold text-transparent bg-clip-text bg-gradient-to-r from-orange-600 to-purple-600': {},
				},
				// '.text-redbow': {
				// 	'@apply font-bold text-transparent bg-clip-text bg-gradient-to-r from-orange-600 to-purple-600': {},
				// },
			};

			addUtilities(newUtilities, ['responsive', 'hover']);
		}),
	],
}
