<script lang="ts">
	import { onMount } from 'svelte';
	import { initWebGPU, type WebGPUStatus } from '$lib/webgpu';
	import { DEFAULT_CONFIG, estimateParams, type ExperimentConfig } from '$lib/model/config';
	import { DataLoader } from '$lib/data/loader';
	import { trainRun, type StepMetrics, type RunResult } from '$lib/train/loop';
	import { sampleText } from '$lib/sample';
	import LossChart from '$lib/components/LossChart.svelte';
	import ConfigEditor from '$lib/components/ConfigEditor.svelte';

	let gpuStatus = $state<WebGPUStatus | null>(null);
	let config = $state<ExperimentConfig>({ ...DEFAULT_CONFIG });
	let running = $state(false);
	let lossData = $state<{ step: number; loss: number }[]>([]);
	let status = $state('idle');
	let result = $state<RunResult | null>(null);
	let sample = $state('');
	let sampling = $state(false);

	let trainLoader: DataLoader | null = null;
	let valLoader: DataLoader | null = null;

	onMount(async () => {
		gpuStatus = await initWebGPU();
		if (gpuStatus.ok) {
			status = 'loading data...';
			[trainLoader, valLoader] = await Promise.all([
				DataLoader.fetch('/data/train.bin'),
				DataLoader.fetch('/data/val.bin')
			]);
			status = 'ready';
		}
	});

	async function startTraining() {
		if (!trainLoader || !valLoader || running) return;

		running = true;
		lossData = [];
		result = null;
		sample = '';
		status = 'training...';

		const r = await trainRun(config, trainLoader, valLoader, {
			onStep(m: StepMetrics) {
				lossData = [...lossData, { step: m.step, loss: m.loss }];
				status = `step ${m.step} | loss ${m.loss.toFixed(4)} | ${(m.elapsed / 1000).toFixed(1)}s`;
			},
			onDone(r: RunResult) {
				result = r;
				status = `done — val_bpb: ${r.valBpb.toFixed(4)} | ${r.totalSteps} steps | ${(r.elapsed / 1000).toFixed(1)}s`;
			}
		});

		running = false;
	}

	async function generateSample() {
		if (!result || sampling) return;
		sampling = true;
		sample = await sampleText(result.params, config, '', 300, 0.8);
		sampling = false;
	}
</script>

<svelte:head>
	<title>autoresearch-webgpu</title>
</svelte:head>

<main class="p-6 max-w-5xl mx-auto space-y-6">
	{#if gpuStatus === null}
		<p class="text-gray-400 font-mono text-sm">initializing webgpu...</p>
	{:else if !gpuStatus.ok}
		<div class="rounded border border-red-800 bg-red-950 p-4 font-mono text-sm text-red-400">
			{gpuStatus.reason}
		</div>
	{:else}
		<div class="grid grid-cols-[280px_1fr] gap-6">
			<!-- Left: config + controls -->
			<div class="space-y-4">
				<div class="rounded border border-gray-800 p-4">
					<h2 class="text-sm font-mono text-gray-400 mb-3">config</h2>
					<ConfigEditor bind:config disabled={running} />
				</div>

				<button
					onclick={startTraining}
					disabled={running || status === 'loading data...'}
					class="w-full rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-4 py-2 font-mono text-sm transition-colors"
				>
					{running ? 'training...' : 'start training'}
				</button>

				{#if result}
					<button
						onclick={generateSample}
						disabled={sampling}
						class="w-full rounded border border-gray-700 hover:border-gray-500 px-4 py-2 font-mono text-sm transition-colors"
					>
						{sampling ? 'generating...' : 'sample text'}
					</button>
				{/if}
			</div>

			<!-- Right: charts + output -->
			<div class="space-y-4">
				<div class="rounded border border-gray-800 p-4">
					<h2 class="text-sm font-mono text-gray-400 mb-2">loss</h2>
					<div class="h-48">
						<LossChart data={lossData} />
					</div>
				</div>

				<div class="rounded border border-gray-800 p-3 font-mono text-sm text-gray-300">
					{status}
				</div>

				{#if result}
					<div class="rounded border border-gray-800 p-4 font-mono text-sm">
						<div class="grid grid-cols-3 gap-2 text-gray-400">
							<div>val_bpb <span class="text-gray-200">{result.valBpb.toFixed(4)}</span></div>
							<div>steps <span class="text-gray-200">{result.totalSteps}</span></div>
							<div>time <span class="text-gray-200">{(result.elapsed / 1000).toFixed(1)}s</span></div>
						</div>
					</div>
				{/if}

				{#if sample}
					<div class="rounded border border-gray-800 p-4">
						<h2 class="text-sm font-mono text-gray-400 mb-2">sample</h2>
						<pre class="text-sm text-gray-300 whitespace-pre-wrap break-all font-mono leading-relaxed max-h-64 overflow-y-auto">{sample}</pre>
					</div>
				{/if}
			</div>
		</div>
	{/if}
</main>
