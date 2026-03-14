<script lang="ts">
	import { onMount } from 'svelte';
	import { initWebGPU, type WebGPUStatus } from '$lib/webgpu';
	import { DEFAULT_CONFIG, type ExperimentConfig } from '$lib/model/config';
	import { DataLoader } from '$lib/data/loader';
	import { trainRun, type StepMetrics, type RunResult } from '$lib/train/loop';
	import { sampleText } from '$lib/sample';
	import { ResearchController } from '$lib/research/controller';
	import type { ExperimentRecord } from '$lib/research/prompt';
	import {
		getDb, insertExperiment, insertInference, insertLossCurve, getAllExperiments,
		getBestExperiment, getInferencesForExperiment, getAllLossCurves, clearAllData,
		exportExperimentsJson, updateWeightsPath, type ExperimentRow, type InferenceRow
	} from '$lib/db';
	import { saveWeights, loadWeights } from '$lib/weights';
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
	let sampling = $state(false);
	let mode = $state<'manual' | 'research'>('manual');
	let experiments = $state<ExperimentRecord[]>([]);
	let currentReasoning = $state('');
	let experimentName = $state('');
	let experimentDesc = $state('');

	// Inference state
	let prompt = $state('');
	let temperature = $state(0.8);
	let selectedExpId = $state<number | null>(null);
	let inferences = $state<InferenceRow[]>([]);
	let inferenceIdx = $state(0);
	let currentExpDbId = $state<number | null>(null);

	let pastLossRuns = $derived(
		experiments
			.filter(e => e.lossCurve && e.lossCurve.length > 1)
			.map(e => ({
				data: e.lossCurve!,
				color: e.id === selectedExpId ? '#ef4444' : e.kept ? '#22c55e' : '#4b5563',
				highlight: e.id === selectedExpId
			}))
	);

	let trainLoader: DataLoader | null = null;
	let valLoader: DataLoader | null = null;
	let controller: ResearchController | null = null;

	onMount(async () => {
		await getDb();
		gpuStatus = await initWebGPU();
		if (gpuStatus.ok) {
			status = 'loading data...';
			[trainLoader, valLoader] = await Promise.all([
				DataLoader.fetch('/data/train.bin'),
				DataLoader.fetch('/data/val.bin')
			]);
			await loadFromDb();
			status = 'ready';
		}
	});

	let lossCurvesMap = $state(new Map<number, { step: number; loss: number }[]>());

	async function loadFromDb() {
		const rows = await getAllExperiments();
		lossCurvesMap = await getAllLossCurves();
		experiments = rows.map(row => ({
			...rowToRecord(row),
			lossCurve: lossCurvesMap.get(row.id) ?? row.loss_curve ?? undefined
		}));
		const best = await getBestExperiment();
		if (best) {
			config = best.config as ExperimentConfig;
		}
	}

	function rowToRecord(row: ExperimentRow): ExperimentRecord {
		return {
			id: row.id,
			name: row.name,
			source: row.source,
			config: row.config as ExperimentConfig,
			valBpb: row.val_bpb,
			elapsed: row.elapsed,
			totalSteps: row.total_steps,
			reasoning: row.reasoning,
			kept: row.kept,
			lossCurve: row.loss_curve ?? undefined
		};
	}

	async function startManualTraining() {
		if (!trainLoader || !valLoader || running) return;

		running = true;
		lossData = [];
		result = null;
		inferences = [];
		inferenceIdx = 0;
		status = 'training...';
		trainLoader.reset();

		const runConfig = { ...config };
		const lossCurve: { step: number; loss: number }[] = [];

		const r = await trainRun(runConfig, trainLoader, valLoader, {
			onStep(m: StepMetrics) {
				lossData = [...lossData, { step: m.step, loss: m.loss }];
				lossCurve.push({ step: m.step, loss: m.loss });
				status = `step ${m.step} | loss ${m.loss.toFixed(4)} | ${(m.elapsed / 1000).toFixed(1)}s`;
			},
			onDone(r: RunResult) {
				result = r;
				status = `done — val_bpb: ${r.valBpb.toFixed(4)} | ${r.totalSteps} steps | ${(r.elapsed / 1000).toFixed(1)}s`;
			}
		});

		const kept = experiments.length === 0 || r.valBpb < Math.min(...experiments.map(e => e.valBpb));

		const dbId = await insertExperiment({
			name: experimentName || `Run ${experiments.length + 1}`,
			source: 'manual',
			config: runConfig,
			valBpb: r.valBpb,
			elapsed: r.elapsed,
			totalSteps: r.totalSteps,
			reasoning: experimentDesc || experimentName || 'Manual run',
			kept,
			lossCurve
		});

		currentExpDbId = dbId;

		// Save loss curve to normalized table
		await insertLossCurve(dbId, lossCurve);

		// Update leaderboard immediately
		await loadFromDb();
		await selectExperimentById(dbId);
		status = `done — val_bpb: ${r.valBpb.toFixed(4)} | ${r.totalSteps} steps`;
		running = false;

		// Save weights + generate sample in background (non-blocking)
		(async () => {
			try {
				const weightsPath = await saveWeights(dbId, r.params);
				await updateWeightsPath(dbId, weightsPath);
			} catch (e) {
				console.error('Failed to save weights:', e);
			}
			try {
				const sampleOutput = await sampleText(r.params, runConfig, '', 200, 0.8);
				await insertInference({ experimentId: dbId, prompt: '', output: sampleOutput, temperature: 0.8 });
				if (selectedExpId === dbId) {
					inferences = await getInferencesForExperiment(dbId);
					inferenceIdx = 0;
				}
			} catch (_) {}
		})();
	}

	async function startResearch() {
		if (!trainLoader || !valLoader || running) return;

		running = true;
		controller = new ResearchController();

		const best = await getBestExperiment();
		if (best) {
			controller.bestConfig = best.config as ExperimentConfig;
			controller.bestBpb = best.val_bpb;
			controller.history = [...experiments];
		}

		await controller.run(trainLoader, valLoader, {
			onExperimentStart(cfg, reasoning) {
				lossData = [];
				currentReasoning = reasoning;
				status = `experiment: ${reasoning}`;
			},
			onStep(m: StepMetrics) {
				lossData = [...lossData, { step: m.step, loss: m.loss }];
			},
			async onExperimentDone(record: ExperimentRecord) {
				await loadFromDb();
				await selectExperimentById(record.id);
				if (record.kept && controller) {
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

	async function selectExperimentById(id: number) {
		selectedExpId = id;
		inferences = await getInferencesForExperiment(id);
		inferenceIdx = 0;
	}

	function selectExperiment(exp: ExperimentRecord) {
		selectExperimentById(exp.id);
	}

	async function generateSample() {
		if (sampling || !selectedExpId) return;
		sampling = true;
		try {
			// Use in-memory params if available, otherwise load from OPFS
			let params;
			let expConfig: ExperimentConfig;
			if (result && currentExpDbId === selectedExpId) {
				params = result.params;
				expConfig = config;
			} else {
				params = await loadWeights(selectedExpId);
				const exp = experiments.find(e => e.id === selectedExpId);
				expConfig = exp?.config ?? config;
			}
			if (!params) {
				console.error('No weights available for experiment', selectedExpId);
				sampling = false;
				return;
			}
			const output = await sampleText(params, expConfig, prompt, 200, temperature);
			await insertInference({ experimentId: selectedExpId, prompt, output, temperature });
			inferences = await getInferencesForExperiment(selectedExpId);
			inferenceIdx = 0;
		} catch (e) {
			console.error('Inference failed:', e);
		}
		sampling = false;
	}

	async function handleClear() {
		if (!confirm('Clear all experiment history?')) return;
		await clearAllData();
		experiments = [];
		config = { ...DEFAULT_CONFIG };
		selectedExpId = null;
		inferences = [];
		currentExpDbId = null;
	}

	async function handleExport() {
		const json = await exportExperimentsJson();
		const blob = new Blob([json], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'autoresearch-experiments.json';
		a.click();
		URL.revokeObjectURL(url);
	}

	let currentInference = $derived(inferences.length > 0 ? inferences[inferenceIdx] : null);
	let selectedExp = $derived(selectedExpId ? experiments.find(e => e.id === selectedExpId) ?? null : null);
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

		<div class="grid grid-cols-[280px_1fr_280px] gap-6">
			<!-- Left: config + controls -->
			<div class="space-y-4">
				<div class="rounded border border-gray-800 p-4">
					<h2 class="text-sm font-mono text-gray-400 mb-3">config</h2>
					<ConfigEditor bind:config disabled={running} />
				</div>

				{#if mode === 'manual'}
					<input
						type="text"
						bind:value={experimentName}
						placeholder="experiment name..."
						disabled={running}
						class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 font-mono text-sm text-gray-200 placeholder-gray-500 disabled:opacity-40"
					/>
					<textarea
						bind:value={experimentDesc}
						placeholder="description / hypothesis..."
						disabled={running}
						rows={2}
						class="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40 resize-none"
					></textarea>
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

			<!-- Center: chart + status + inference (all in one panel) -->
			<div class="space-y-4">
				<div class="rounded border border-gray-800 p-4 space-y-4">
					{#if selectedExp}
						<div class="flex items-center gap-2 font-mono text-sm">
							<span class="px-1.5 py-0.5 rounded text-xs {selectedExp.source === 'auto' ? 'bg-blue-900/50 text-blue-300' : 'bg-gray-700 text-gray-300'}">
								{selectedExp.source === 'auto' ? 'auto' : 'manual'}
							</span>
							<span class="text-gray-200">{selectedExp.name}</span>
							<span class="text-gray-500 tabular-nums ml-auto">{selectedExp.valBpb.toFixed(4)} bpb</span>
						</div>
						{#if selectedExp.reasoning && selectedExp.reasoning !== selectedExp.name}
							<p class="text-xs text-gray-400 font-mono">{selectedExp.reasoning}</p>
						{/if}
					{:else}
						<h2 class="text-sm font-mono text-gray-400">loss</h2>
					{/if}
					<div class="h-48">
						<LossChart data={lossData} pastRuns={pastLossRuns} />
					</div>

					<div class="font-mono text-sm text-gray-300">
						{status}
					</div>

					<!-- Inference inline -->
					<div class="border-t border-gray-800 pt-3 space-y-2">
						<div class="flex items-center gap-2">
							<input
								type="text"
								bind:value={prompt}
								placeholder="prompt..."
								disabled={!selectedExpId || sampling}
								class="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
								onkeydown={(e: KeyboardEvent) => { if (e.key === 'Enter') generateSample(); }}
							/>
							<input
								type="number"
								bind:value={temperature}
								min={0.1}
								max={2}
								step={0.1}
								disabled={!selectedExpId || sampling}
								class="w-12 bg-gray-800 border border-gray-700 rounded px-1 py-1 text-right tabular-nums text-xs text-gray-200 font-mono disabled:opacity-40"
								title="temperature"
							/>
							<button
								onclick={generateSample}
								disabled={!selectedExpId || sampling}
								class="rounded bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 px-2 py-1 font-mono text-xs transition-colors"
							>
								{sampling ? '...' : 'go'}
							</button>
						</div>
						{#if currentInference}
							<div class="flex items-center justify-between text-xs font-mono text-gray-500">
								<span>
									{#if currentInference.prompt}"{currentInference.prompt}"{:else}(empty prompt){/if}
									· t={currentInference.temperature}
								</span>
								{#if inferences.length > 1}
									<div class="flex items-center gap-1">
										<button
											onclick={() => { inferenceIdx = Math.min(inferenceIdx + 1, inferences.length - 1); }}
											disabled={inferenceIdx >= inferences.length - 1}
											class="px-1 hover:text-gray-300 disabled:opacity-30"
										>←</button>
										<span>{inferences.length - inferenceIdx}/{inferences.length}</span>
										<button
											onclick={() => { inferenceIdx = Math.max(inferenceIdx - 1, 0); }}
											disabled={inferenceIdx <= 0}
											class="px-1 hover:text-gray-300 disabled:opacity-30"
										>→</button>
									</div>
								{/if}
							</div>
							<pre class="text-xs text-gray-300 whitespace-pre-wrap break-all font-mono leading-relaxed max-h-48 overflow-y-auto">{currentInference.output}</pre>
						{:else if !selectedExpId}
							<p class="text-gray-500 text-xs font-mono">train a model to generate text</p>
						{/if}
					</div>
				</div>

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
					<Leaderboard {experiments} onSelect={selectExperiment} selected={selectedExpId ? experiments.find(e => e.id === selectedExpId) ?? null : null} />
				</div>
			</div>
		</div>
	{/if}
</main>
