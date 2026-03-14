<script lang="ts">
	import { onMount } from 'svelte';
	import { initWebGPU, type WebGPUStatus } from '$lib/webgpu';

	let status = $state<WebGPUStatus | null>(null);

	onMount(async () => {
		status = await initWebGPU();
	});
</script>

<svelte:head>
	<title>autoresearch-webgpu</title>
</svelte:head>

<main class="p-6 max-w-4xl mx-auto">
	{#if status === null}
		<p class="text-gray-400 font-mono text-sm">Initializing WebGPU...</p>
	{:else if status.ok}
		<div class="space-y-4">
			<div class="rounded border border-green-800 bg-green-950 p-4 font-mono text-sm">
				<p class="text-green-400">WebGPU ready.</p>
				<p class="text-gray-400 mt-1">
					Adapter: {status.adapterInfo.vendor} — {status.adapterInfo.architecture}
				</p>
			</div>
			<p class="text-gray-400 text-sm">Training engine loaded. Next: model + data.</p>
		</div>
	{:else}
		<div class="rounded border border-red-800 bg-red-950 p-4 font-mono text-sm">
			<p class="text-red-400">{status.reason}</p>
		</div>
	{/if}
</main>
