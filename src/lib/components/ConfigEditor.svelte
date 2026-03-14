<script lang="ts">
	import { DEFAULT_CONFIG, estimateParams, type ExperimentConfig } from '$lib/model/config';

	let { config = $bindable(), disabled = false }: { config: ExperimentConfig; disabled?: boolean } =
		$props();

	let paramCount = $derived(estimateParams(config));
</script>

<div class="space-y-3 text-sm font-mono">
	<div class="text-gray-400">
		~{(paramCount / 1e6).toFixed(2)}M params
	</div>

	<div class="grid grid-cols-2 gap-2">
		<label class="flex justify-between">
			<span class="text-gray-400">layers</span>
			<input type="number" bind:value={config.nLayer} min={1} max={12} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">d_model</span>
			<input type="number" bind:value={config.nEmbd} min={32} max={512} step={32} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">heads</span>
			<input type="number" bind:value={config.nHead} min={1} max={16} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">mlp_ratio</span>
			<input type="number" bind:value={config.mlpRatio} min={1} max={8} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">activation</span>
			<select bind:value={config.activation} {disabled} class="bg-gray-800 border border-gray-700 rounded px-1">
				<option value="relu_sq">relu²</option>
				<option value="gelu">gelu</option>
				<option value="silu">silu</option>
			</select>
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">seq_len</span>
			<input type="number" bind:value={config.seqLen} min={32} max={512} step={32} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">lr</span>
			<input type="number" bind:value={config.lr} min={0.00001} max={0.01} step={0.0001} {disabled} class="w-20 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">batch_size</span>
			<input type="number" bind:value={config.batchSize} min={1} max={64} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">train_sec</span>
			<input type="number" bind:value={config.trainSeconds} min={10} max={300} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>

		<label class="flex justify-between">
			<span class="text-gray-400">weight_decay</span>
			<input type="number" bind:value={config.weightDecay} min={0} max={1} step={0.01} {disabled} class="w-16 bg-gray-800 border border-gray-700 rounded px-1 text-right" />
		</label>
	</div>

	<button
		onclick={() => (config = { ...DEFAULT_CONFIG })}
		{disabled}
		class="text-xs text-gray-500 hover:text-gray-300"
	>
		reset defaults
	</button>
</div>
