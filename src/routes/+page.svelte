<script lang="ts">
	import { onMount } from 'svelte';
	import { initWebGPU, type WebGPUStatus } from '$lib/webgpu';
	import { DEFAULT_CONFIG, estimateParams, type ExperimentConfig } from '$lib/model/config';
	import { DataLoader } from '$lib/data/loader';
	import { trainRun, type StepMetrics, type RunResult } from '$lib/train/loop';
	import { sampleText } from '$lib/sample';
	import { ResearchController } from '$lib/research/controller';
	import type { ExperimentRecord } from '$lib/research/prompt';
	import { saveExperiment, loadExperiments, saveBestConfig, loadBestConfig, clearAll, exportAsJson } from '$lib/storage';
	import LossChart from '$lib/components/LossChart.svelte';
	import ConfigEditor from '$lib/components/ConfigEditor.svelte';
	import ResearchLog from '$lib/components/ResearchLog.svelte';
	import Leaderboard from '$lib/components/Leaderboard.svelte';

	let gpuStatus = $state<WebGPUStatus | null>(null);
	let config = $state<ExperimentConfig>({ ...DEFAULT_CONFIG });
	let running = $state(false);
	let lossData = $state<{ step: number; loss: number }[]>([]);
	let status = $state('idle');
	let result = $state<RunResult | null>(null);
	let sample = $state('');
	let sampling = $state(false);
	let mode = $state<'manual' | 'research'>('manual');
	let experiments = $state<ExperimentRecord[]>([]);
	let currentReasoning = $state('');
	let prompt = $state('');
	let temperature = $state(0.8);
	let pastLossRuns = $derived(
		experiments
			.filter(e => e.lossCurve && e.lossCurve.length > 1)
			.map(e => ({
				data: e.lossCurve!,
				color: e.kept ? '#22c55e' : '#6b7280'
			}))
	);

	let trainLoader: DataLoader | null = null;
	let valLoader: DataLoader | null = null;
	let controller: ResearchController | null = null;

	onMount(async () => {
		gpuStatus = await initWebGPU();
		if (gpuStatus.ok) {
			status = 'loading data...';
			[trainLoader, valLoader] = await Promise.all([
				DataLoader.fetch('/data/train.bin'),
				DataLoader.fetch('/data/val.bin')
			]);

			// Restore previous session
			const saved = await loadExperiments();
			if (saved.length > 0) {
				experiments = saved;
			}
			const best = await loadBestConfig();
			if (best) {
				config = best.config;
			}

			status = 'ready';
		}
	});

	async function startManualTraining() {
		if (!trainLoader || !valLoader || running) return;

		running = true;
		lossData = [];
		result = null;
		sample = '';
		status = 'training...';
		trainLoader.reset();

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

	async function startResearch() {
		if (!trainLoader || !valLoader || running) return;

		running = true;
		controller = new ResearchController();

		// Restore best if we have history
		const best = await loadBestConfig();
		if (best) {
			controller.bestConfig = best.config;
			controller.bestBpb = best.bpb;
			controller.history = [...experiments];
		}

		await controller.run(trainLoader, valLoader, {
			onExperimentStart(id, config, reasoning) {
				lossData = [];
				currentReasoning = reasoning;
				status = `experiment #${id}: ${reasoning}`;
			},
			onStep(m: StepMetrics) {
				lossData = [...lossData, { step: m.step, loss: m.loss }];
			},
			async onExperimentDone(record: ExperimentRecord) {
				experiments = [...experiments, record];
				await saveExperiment(record);
				if (record.kept && controller) {
					await saveBestConfig(controller.bestConfig, controller.bestBpb);
					config = { ...controller.bestConfig };
				}
				status = `#${record.id} ${record.kept ? 'KEPT' : 'discarded'} — bpb ${record.valBpb.toFixed(4)}`;
			},
			onError(error) {
				status = `error: ${error}`;
			}
		});

		running = false;
	}

	function stopResearch() {
		controller?.stop();
	}

	async function generateSample() {
		if (!result || sampling) return;
		sampling = true;
		sample = await sampleText(result.params, config, prompt, 200, temperature);
		sampling = false;
	}

	async function handleClear() {
		if (!confirm('Clear all experiment history?')) return;
		await clearAll();
		experiments = [];
		config = { ...DEFAULT_CONFIG };
	}

	function handleExport() {
		const blob = new Blob([exportAsJson(experiments)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'autoresearch-experiments.json';
		a.click();
		URL.revokeObjectURL(url);
	}
</script>

<svelte:head>
	<title>autoresearch-webgpu</title>
</svelte:head>

<main class="p-6 max-w-6xl mx-auto space-y-6">
	{#if gpuStatus === null}
		<p class="text-gray-400 font-mono text-sm">initializing webgpu...</p>
	{:else if !gpuStatus.ok}
		<div class="rounded border border-red-800 bg-red-950 p-4 font-mono text-sm text-red-400">
			{gpuStatus.reason}
		</div>
	{:else}
		<!-- Mode toggle -->
		<div class="flex items-center gap-4">
			<div class="flex rounded border border-gray-700 text-sm font-mono overflow-hidden">
				<button
					class="px-3 py-1 {mode === 'manual' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}"
					onclick={() => (mode = 'manual')}
					disabled={running}
				>
					manual
				</button>
				<button
					class="px-3 py-1 {mode === 'research' ? 'bg-blue-700 text-white' : 'text-gray-400 hover:text-white'}"
					onclick={() => (mode = 'research')}
					disabled={running}
				>
					research
				</button>
			</div>
			<span class="text-gray-500 text-xs font-mono">
				{experiments.length} experiments
			</span>
			{#if experiments.length > 0}
				<button onclick={handleExport} class="text-gray-500 hover:text-gray-300 text-xs font-mono">
					export
				</button>
				<button onclick={handleClear} disabled={running} class="text-gray-500 hover:text-red-400 text-xs font-mono">
					clear
				</button>
			{/if}
		</div>

		<div class="grid grid-cols-[280px_1fr_240px] gap-6">
			<!-- Left: config + controls -->
			<div class="space-y-4">
				<div class="rounded border border-gray-800 p-4">
					<h2 class="text-sm font-mono text-gray-400 mb-3">config</h2>
					<ConfigEditor bind:config disabled={running} />
				</div>

				{#if mode === 'manual'}
					<button
						onclick={startManualTraining}
						disabled={running || status === 'loading data...'}
						class="w-full rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-4 py-2 font-mono text-sm transition-colors"
					>
						{running ? 'training...' : 'start training'}
					</button>
				{:else}
					{#if !running}
						<button
							onclick={startResearch}
							disabled={status === 'loading data...'}
							class="w-full rounded bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:text-gray-500 px-4 py-2 font-mono text-sm transition-colors"
						>
							start research
						</button>
					{:else}
						<button
							onclick={stopResearch}
							class="w-full rounded bg-red-600 hover:bg-red-500 px-4 py-2 font-mono text-sm transition-colors"
						>
							stop after current run
						</button>
					{/if}
				{/if}
			</div>

			<!-- Center: charts + output -->
			<div class="space-y-4">
				<div class="rounded border border-gray-800 p-4">
					<h2 class="text-sm font-mono text-gray-400 mb-2">loss</h2>
					<div class="h-48">
						<LossChart data={lossData} pastRuns={pastLossRuns} />
					</div>
				</div>

				<div class="rounded border border-gray-800 p-3 font-mono text-sm text-gray-300">
					{status}
				</div>

				{#if result && mode === 'manual'}
					<div class="rounded border border-gray-800 p-4 font-mono text-sm">
						<div class="grid grid-cols-3 gap-2 text-gray-400">
							<div>val_bpb <span class="text-gray-200">{result.valBpb.toFixed(4)}</span></div>
							<div>steps <span class="text-gray-200">{result.totalSteps}</span></div>
							<div>time <span class="text-gray-200">{(result.elapsed / 1000).toFixed(1)}s</span></div>
						</div>
					</div>

					<div class="rounded border border-gray-800 p-4 space-y-3">
						<h2 class="text-sm font-mono text-gray-400">inference</h2>
						<div class="flex gap-2">
							<input
								type="text"
								bind:value={prompt}
								placeholder="enter a prompt..."
								class="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-1.5 font-mono text-sm text-gray-200 placeholder-gray-500"
								onkeydown={(e: KeyboardEvent) => { if (e.key === 'Enter') generateSample(); }}
							/>
							<label class="flex items-center gap-1 text-xs text-gray-500 font-mono">
								temp
								<input type="number" bind:value={temperature} min={0.1} max={2} step={0.1} class="w-14 bg-gray-800 border border-gray-700 rounded px-1 py-1.5 text-right tabular-nums text-sm text-gray-200" />
							</label>
							<button
								onclick={generateSample}
								disabled={sampling}
								class="rounded bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 px-3 py-1.5 font-mono text-sm transition-colors"
							>
								{sampling ? '...' : 'generate'}
							</button>
						</div>
						{#if sample}
							<pre class="text-sm text-gray-300 whitespace-pre-wrap break-all font-mono leading-relaxed max-h-48 overflow-y-auto">{sample}</pre>
						{/if}
					</div>
				{/if}

				{#if mode === 'research' && experiments.length > 0}
					<div class="rounded border border-gray-800 p-4">
						<h2 class="text-sm font-mono text-gray-400 mb-2">research log</h2>
						<ResearchLog {experiments} bestConfig={config} />
					</div>
				{/if}
			</div>

			<!-- Right: leaderboard -->
			<div class="space-y-4">
				<div class="rounded border border-gray-800 p-4">
					<h2 class="text-sm font-mono text-gray-400 mb-3">leaderboard</h2>
					<Leaderboard {experiments} />
				</div>
			</div>
		</div>
	{/if}
</main>
