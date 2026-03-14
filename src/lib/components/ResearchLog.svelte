<script lang="ts">
	import type { ExperimentRecord } from '$lib/research/prompt';
	import type { ExperimentConfig } from '$lib/model/config';
	import ConfigDiff from './ConfigDiff.svelte';

	let { experiments, bestConfig, onSelect, selectedId }: { experiments: ExperimentRecord[]; bestConfig: ExperimentConfig; onSelect?: (exp: ExperimentRecord) => void; selectedId?: number | null } = $props();
</script>

<div class="space-y-2 max-h-96 overflow-y-auto">
	{#each [...experiments].reverse() as exp}
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			onclick={() => onSelect?.(exp)}
			class="rounded border px-3 py-2 text-sm font-mono cursor-pointer transition-colors {selectedId === exp.id ? 'border-blue-700 bg-blue-950/30' : exp.kept ? 'border-green-800 bg-green-950/30 hover:bg-green-950/50' : 'border-gray-800 hover:bg-gray-800/50'}"
		>
			<div class="flex items-center justify-between">
				<span class="text-gray-400">#{exp.id}</span>
				<span class="text-xs {exp.kept ? 'text-green-400' : 'text-gray-500'}">
					{exp.kept ? 'KEPT' : 'discarded'} — bpb {exp.valBpb.toFixed(4)}
				</span>
			</div>
			<p class="text-gray-300 text-xs mt-1">{exp.reasoning}</p>
			<div class="mt-1">
				<ConfigDiff before={bestConfig} after={exp.config} />
			</div>
		</div>
	{/each}

	{#if experiments.length === 0}
		<p class="text-gray-500 text-sm font-mono">no experiments yet</p>
	{/if}
</div>
