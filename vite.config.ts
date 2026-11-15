import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	optimizeDeps: {
		exclude: ['@electric-sql/pglite']
	},
	// Served behind a proxy under another host name, so accept any Host header.
	server: {
		allowedHosts: true
	},
	preview: {
		allowedHosts: true
	}
});
