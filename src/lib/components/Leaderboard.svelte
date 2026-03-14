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
			class="w-full flex items-center gap-2 font-mono text-xs px-2 py-1.5 rounded transition-colors
				{selected?.id === exp.id ? 'bg-blue-950/50 text-blue-300' : i === 0 ? 'bg-green-950/50 text-green-300' : 'text-gray-400 hover:bg-gray-800'}"
		>
			<span class="shrink-0 w-3 text-center {exp.source === 'auto' ? 'text-blue-400' : 'text-gray-500'}" title={exp.source === 'auto' ? 'auto (Claude)' : 'manual'}>
				{exp.source === 'auto' ? 'A' : 'M'}
			</span>
			<span class="truncate text-left flex-1" title={exp.reasoning}>
				{exp.name || `#${exp.id}`}
			</span>
			<span class="tabular-nums shrink-0">{exp.valBpb.toFixed(3)}</span>
		</button>
	{/each}

	{#if experiments.length === 0}
		<p class="text-gray-500 text-xs font-mono px-2">no experiments yet</p>
	{/if}
</div>
