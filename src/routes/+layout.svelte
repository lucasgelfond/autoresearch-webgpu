<script lang="ts">
	import '../app.css';

	let { children } = $props();
</script>

<div class="min-h-screen bg-gray-950 text-gray-100">
	<header class="border-b border-gray-800 px-6 py-3">
		<h1 class="text-lg font-mono tracking-tight">autoresearch-webgpu</h1>
	</header>
	{@render children()}
</div>
