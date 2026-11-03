<script lang="ts">
	import type { ExperimentRecord } from '$lib/research/prompt';

	let {
		experiments,
		onSelect,
		selected,
		sortByLoss = true
	}: {
		experiments: ExperimentRecord[];
		onSelect?: (exp: ExperimentRecord) => void;
		selected?: ExperimentRecord | null;
		sortByLoss?: boolean;
	} = $props();

	let sorted = $derived(
		sortByLoss
			? [...experiments].sort((a, b) => a.valBpb - b.valBpb)
			: [...experiments].reverse()
	);
	let bestId = $derived(
		experiments.length > 0
			? [...experiments].sort((a, b) => a.valBpb - b.valBpb)[0].id
			: null
	);
</script>

<div class="space-y-0.5 overflow-y-auto">
	{#each sorted as exp, i}
		<button
			onclick={() => onSelect?.(exp)}
			class="w-full flex items-center gap-1.5 font-mono text-[11px] px-1.5 py-1 rounded transition-colors
				{exp.id === -1 ? 'bg-red-950/50 text-red-300' : selected?.id === exp.id ? 'bg-blue-950/50 text-blue-300' : exp.id === bestId ? 'bg-green-950/50 text-green-300' : 'text-gray-400 hover:bg-gray-800'}"
		>
			{#if exp.id === -1}
				<span class="shrink-0 w-3 text-center text-red-400 animate-pulse" title="in progress">*</span>
			{:else}
				<span class="shrink-0 w-3 text-center {exp.source === 'auto' ? 'text-blue-400' : 'text-gray-500'}" title={exp.source === 'auto' ? 'auto (model)' : 'manual'}>
					{exp.source === 'auto' ? 'A' : 'M'}
				</span>
			{/if}
			<span class="truncate text-left flex-1" title={exp.reasoning}>
				{exp.name || `#${exp.id}`}
			</span>
			<span class="tabular-nums shrink-0">{exp.id === -1 && exp.valBpb === Infinity ? '...' : exp.valBpb.toFixed(3)}</span>
		</button>
	{/each}

	{#if experiments.length === 0}
		<p class="text-gray-500 text-xs font-mono px-2">no experiments yet</p>
	{/if}
</div>
