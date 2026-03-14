<script lang="ts">
	import type { ExperimentRecord } from '$lib/research/prompt';

	let {
		experiments,
		onSelect,
		selected
	}: {
		experiments: ExperimentRecord[];
		onSelect?: (exp: ExperimentRecord) => void;
		selected?: ExperimentRecord | null;
	} = $props();

	let sorted = $derived([...experiments].sort((a, b) => a.valBpb - b.valBpb).slice(0, 10));
</script>

<div class="space-y-1">
	{#each sorted as exp, i}
		<button
			onclick={() => onSelect?.(exp)}
			class="w-full flex items-center justify-between font-mono text-xs px-2 py-1 rounded transition-colors
				{selected?.id === exp.id ? 'bg-blue-950/50 text-blue-300' : i === 0 ? 'bg-green-950/50 text-green-300' : 'text-gray-400 hover:bg-gray-800'}"
		>
			<span>#{exp.id}</span>
			<span class="tabular-nums">{exp.valBpb.toFixed(4)}</span>
			<span class="text-gray-500">{exp.totalSteps}st</span>
		</button>
	{/each}

	{#if experiments.length === 0}
		<p class="text-gray-500 text-xs font-mono px-2">no experiments yet</p>
	{/if}
</div>
