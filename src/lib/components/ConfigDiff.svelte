<script lang="ts">
	import type { ExperimentConfig } from '$lib/model/config';

	let { before, after }: { before: ExperimentConfig; after: ExperimentConfig } = $props();

	type Diff = { key: string; from: string; to: string };

	let diffs = $derived.by(() => {
		const result: Diff[] = [];
		for (const key of Object.keys(before) as (keyof ExperimentConfig)[]) {
			if (before[key] !== after[key]) {
				result.push({ key, from: String(before[key]), to: String(after[key]) });
			}
		}
		return result;
	});
</script>

{#if diffs.length === 0}
	<span class="text-gray-500 text-xs">no changes</span>
{:else}
	<div class="flex flex-wrap gap-1">
		{#each diffs as d}
			<span class="inline-flex items-center gap-1 text-xs rounded bg-gray-800 px-1.5 py-0.5">
				<span class="text-gray-400">{d.key}</span>
				<span class="text-red-400 line-through">{d.from}</span>
				<span class="text-green-400">{d.to}</span>
			</span>
		{/each}
	</div>
{/if}
