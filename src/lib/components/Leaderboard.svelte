<script lang="ts">
	import type { ExperimentRecord } from '$lib/research/prompt';

	let {
		experiments,
		onSelect,
		selected,
		sortMode = 'bpb',
		selectionEnabled = false,
		selectedIds = [],
		onToggleBatchSelect
	}: {
		experiments: ExperimentRecord[];
		onSelect?: (exp: ExperimentRecord) => void;
		selected?: ExperimentRecord | null;
		sortMode?: 'bpb' | 'newest' | 'oldest' | 'steps' | 'name';
		selectionEnabled?: boolean;
		selectedIds?: number[];
		onToggleBatchSelect?: (expId: number) => void;
	} = $props();

	let dataGridTemplate = '14px 44px minmax(0,1fr) 12px 28px 44px';

	let sorted = $derived.by(() => {
		const items = [...experiments];
		switch (sortMode) {
			case 'name':
				return items.sort(
					(a, b) =>
						(a.name || '').localeCompare(b.name || '', undefined, { sensitivity: 'base' }) ||
						b.id - a.id
				);
			case 'newest':
				return items.sort((a, b) => {
					const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
					const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
					const byTime = (Number.isFinite(bTime) ? bTime : b.id) - (Number.isFinite(aTime) ? aTime : a.id);
					return byTime || b.id - a.id;
				});
			case 'oldest':
				return items.sort((a, b) => {
					const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
					const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
					const byTime = (Number.isFinite(aTime) ? aTime : a.id) - (Number.isFinite(bTime) ? bTime : b.id);
					return byTime || a.id - b.id;
				});
			case 'steps':
				return items.sort((a, b) => b.totalSteps - a.totalSteps || a.valBpb - b.valBpb);
			case 'bpb':
			default:
				return items.sort((a, b) => a.valBpb - b.valBpb || b.id - a.id);
		}
	});
	let bestId = $derived(
		experiments.length > 0
			? [...experiments].sort((a, b) => a.valBpb - b.valBpb)[0].id
			: null
	);
</script>

<div class="space-y-0.5 overflow-y-auto">
	<div class="sticky top-0 z-10 flex items-center gap-1.5 border-b border-gray-800 bg-gray-950/95 px-1.5 py-1 font-mono text-[9px] uppercase tracking-[0.14em] text-gray-500 backdrop-blur">
		{#if selectionEnabled}
			<span class="ml-1 shrink-0 w-3" aria-hidden="true"></span>
		{/if}
		<div class="min-w-0 flex-1 grid items-center gap-1.5" style={`grid-template-columns: ${dataGridTemplate};`}>
			<span class="text-center" title="run source">src</span>
			<span class="tabular-nums text-right" title="experiment id">run</span>
			<span class="min-w-0" title="experiment name">experiment</span>
			<span class="text-center" title="rerun marker">r</span>
			<span class="tabular-nums text-right" title="training steps">steps</span>
			<span class="tabular-nums text-right" title="validation bits per byte">bpb</span>
		</div>
	</div>
	{#each sorted as exp, i}
		<div
			class="w-full flex items-center gap-1.5 font-mono text-[11px] rounded transition-colors
				{exp.id === -1 ? 'bg-red-950/50 text-red-300' : selected?.id === exp.id ? 'bg-blue-950/50 text-blue-300' : exp.id === bestId ? 'bg-green-950/50 text-green-300' : 'text-gray-400 hover:bg-gray-800'}"
		>
			{#if selectionEnabled && exp.id !== -1}
				<button
					type="button"
					class="ml-1 shrink-0 w-3 h-3 rounded border border-gray-600 text-[9px] leading-none flex items-center justify-center {selectedIds.includes(exp.id) ? 'bg-blue-600 border-blue-500 text-white' : 'text-transparent'}"
					onclick={() => {
						onToggleBatchSelect?.(exp.id);
					}}
					title={selectedIds.includes(exp.id) ? 'remove from rerun selection' : 'add to rerun selection'}
				>
					✓
				</button>
			{/if}
			<button
				onclick={() => onSelect?.(exp)}
				class="min-w-0 flex-1 grid items-center gap-1.5 px-1.5 py-1 text-left"
				style={`grid-template-columns: ${dataGridTemplate};`}
			>
			{#if exp.id === -1}
				<span class="text-center text-red-400 animate-pulse" title="in progress">*</span>
			{:else}
				<span class="text-center {exp.source === 'auto' ? 'text-blue-400' : 'text-gray-500'}" title={exp.source === 'auto' ? 'automatic experiment' : 'manual experiment'}>
					{exp.source === 'auto' ? 'A' : 'M'}
				</span>
			{/if}
			{#if exp.id !== -1}
				<span class="tabular-nums text-[9px] text-right text-gray-500" title={`run #${exp.id}`}>#{exp.id}</span>
			{:else}
				<span class="tabular-nums text-[9px] text-right text-gray-500" aria-hidden="true">...</span>
			{/if}
			<span class="truncate text-left flex-1" title={exp.reasoning}>
				{exp.name || `#${exp.id}`}
			</span>
			{#if exp.rerunOf}
				<span class="text-center text-[9px] text-amber-400" title={`rerun of #${exp.rerunOf}`}>R</span>
			{:else}
				<span class="text-center" aria-hidden="true"></span>
			{/if}
			<span class="tabular-nums text-[9px] text-right text-gray-500" title={`${exp.totalSteps} steps`}>
				{exp.id === -1 ? '...' : exp.totalSteps}
			</span>
			<span class="tabular-nums text-right">{exp.id === -1 && exp.valBpb === Infinity ? '...' : exp.valBpb.toFixed(3)}</span>
			</button>
		</div>
	{/each}

	{#if experiments.length === 0}
		<p class="text-gray-500 text-xs font-mono px-2">no experiments yet</p>
	{/if}
</div>
