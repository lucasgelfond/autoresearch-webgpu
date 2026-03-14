<script lang="ts">
	import { onMount } from 'svelte';
	import { initWebGPU, type WebGPUStatus } from '$lib/webgpu';
	import { DEFAULT_CONFIG, DEFAULT_CONSTRAINTS, type ExperimentConfig, type ParamConstraints } from '$lib/model/config';
	import { DataLoader } from '$lib/data/loader';
	import { trainRun, type StepMetrics, type RunResult } from '$lib/train/loop';
	import { sampleText } from '$lib/sample';
	import { ResearchController } from '$lib/research/controller';
	import type { ExperimentRecord } from '$lib/research/prompt';
	import {
		getDb, insertExperiment, insertInference, insertLossCurve, getAllExperiments,
		getBestExperiment, getInferencesForExperiment, getAllLossCurves, clearAllData,
		exportCsvZip, updateWeightsPath, rowToConfig, type ExperimentRow, type InferenceRow
	} from '$lib/db';
	import { estimateParams } from '$lib/model/config';
	import { saveWeights, loadWeights } from '$lib/weights';
	import LossChart from '$lib/components/LossChart.svelte';
	import ConfigEditor from '$lib/components/ConfigEditor.svelte';
	import Leaderboard from '$lib/components/Leaderboard.svelte';
	import ConfigDiff from '$lib/components/ConfigDiff.svelte';
	import ConstraintsModal from '$lib/components/ConstraintsModal.svelte';

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
	let listMode = $state<'leaderboard' | 'current'>('leaderboard');

	// Inference state
	let prompt = $state('');
	let temperature = $state(0.8);
	let selectedExpId = $state<number | null>(null);
	let inferences = $state<InferenceRow[]>([]);
	let inferenceIdx = $state(0);
	let currentExpDbId = $state<number | null>(null);
	let viewingLiveRun = $state(true);
	let currentRunName = $state('');
	let trainAbort: AbortController | null = null;
	let inProgressExp = $state<ExperimentRecord | null>(null);
	let waitingForRecommendation = $state(false);

	let allExperiments = $derived(
		inProgressExp ? [...experiments, inProgressExp] : experiments
	);

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
			// Restore state from URL
			const params = new URL(window.location.href).searchParams;
			const expParam = params.get('exp');
			if (expParam) {
				const id = Number(expParam);
				const exp = experiments.find(e => e.id === id);
				if (exp) selectExperiment(exp);
			}
			const viewParam = params.get('view');
			if (viewParam === 'current' || viewParam === 'leaderboard') listMode = viewParam;
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
			config = rowToConfig(best);
		}
	}

	function rowToRecord(row: ExperimentRow): ExperimentRecord {
		return {
			id: row.id,
			name: row.name,
			source: row.source,
			config: rowToConfig(row),
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
		currentRunName = experimentName || `Run ${experiments.length + 1}`;
		currentReasoning = experimentDesc || '';
		trainLoader.reset();
		trainAbort = new AbortController();
		setListMode('current');

		const runConfig = { ...config };
		const lossCurve: { step: number; loss: number }[] = [];

		// Create in-progress experiment for leaderboard
		inProgressExp = {
			id: -1,
			name: currentRunName,
			source: 'manual',
			config: runConfig,
			valBpb: Infinity,
			elapsed: 0,
			totalSteps: 0,
			reasoning: currentReasoning,
			kept: false,
		};

		const r = await trainRun(runConfig, trainLoader, valLoader, {
			signal: trainAbort.signal,
			onStep(m: StepMetrics) {
				lossData = [...lossData, { step: m.step, loss: m.loss }];
				lossCurve.push({ step: m.step, loss: m.loss });
				status = `step ${m.step} | loss ${m.loss.toFixed(4)} | ${(m.elapsed / 1000).toFixed(1)}s`;
				// Update in-progress experiment with latest loss
				if (inProgressExp) {
					inProgressExp = { ...inProgressExp, valBpb: m.loss, totalSteps: m.step, elapsed: m.elapsed };
				}
			},
			onDone(r: RunResult) {
				result = r;
				status = `done — val_bpb: ${r.valBpb.toFixed(4)} | ${r.totalSteps} steps | ${(r.elapsed / 1000).toFixed(1)}s`;
			}
		});
		trainAbort = null;
		inProgressExp = null;

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
		waitingForRecommendation = true;
		lossData = [];
		status = 'starting...';
		currentRunName = '';
		currentReasoning = '';
		controller = new ResearchController();
		controller.constraints = constraints;
		setListMode('current');

		const best = await getBestExperiment();
		if (best) {
			controller.bestConfig = rowToConfig(best);
			controller.bestBpb = best.val_bpb;
			controller.history = [...experiments];
		}

		await controller.run(trainLoader, valLoader, {
			onExperimentStart(cfg, reasoning) {
				waitingForRecommendation = false;
				lossData = [];
				currentReasoning = reasoning;
				currentRunName = `Research #${(controller?.history.length ?? 0) + 1}`;
				status = `experiment: ${reasoning}`;
				if (listMode === 'current') setSelectedExp(null);
				// Create in-progress experiment
				inProgressExp = {
					id: -1,
					name: currentRunName,
					source: 'auto',
					config: cfg,
					valBpb: Infinity,
					elapsed: 0,
					totalSteps: 0,
					reasoning,
					kept: false,
				};
			},
			onStep(m: StepMetrics) {
				lossData = [...lossData, { step: m.step, loss: m.loss }];
				if (inProgressExp) {
					inProgressExp = { ...inProgressExp, valBpb: m.loss, totalSteps: m.step, elapsed: m.elapsed };
				}
			},
			async onExperimentDone(record: ExperimentRecord) {
				inProgressExp = null;
				waitingForRecommendation = true;
				await loadFromDb();
				if (listMode === 'current') setSelectedExp(null);
				if (record.kept && controller) {
					config = { ...controller.bestConfig };
				}
				status = `#${record.id} ${record.kept ? 'KEPT' : 'discarded'} — bpb ${record.valBpb.toFixed(4)}`;
			},
			onError(error) {
				status = `error: ${error}`;
			}
		});

		inProgressExp = null;
		waitingForRecommendation = false;
		running = false;
	}

	function stopCurrentRun() {
		trainAbort?.abort();
		controller?.stop();
		controller?.stopCurrentRun();
	}

	function setSelectedExp(id: number | null) {
		selectedExpId = id;
		const url = new URL(window.location.href);
		if (id != null) {
			url.searchParams.set('exp', String(id));
		} else {
			url.searchParams.delete('exp');
		}
		history.replaceState(null, '', url);
	}

	function setListMode(m: 'leaderboard' | 'current') {
		listMode = m;
		if (m === 'current') {
			// Clear selected experiment to show the live run
			setSelectedExp(null);
		}
		const url = new URL(window.location.href);
		url.searchParams.set('view', m);
		history.replaceState(null, '', url);
	}

	function selectExperimentById(id: number) {
		setSelectedExp(id);
		inferenceIdx = 0;
		// Load inferences in background — don't block the UI
		getInferencesForExperiment(id).then(rows => {
			if (selectedExpId === id) {
				inferences = rows;
			}
		});
	}

	function selectExperiment(exp: ExperimentRecord) {
		config = { ...exp.config };
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

	let showClearModal = $state(false);
	let showConstraints = $state(false);
	let constraints = $state<ParamConstraints>({ ...DEFAULT_CONSTRAINTS });
	let maxParams = $state(400_000_000);

	async function handleClear() {
		showClearModal = true;
	}

	async function confirmClear() {
		showClearModal = false;
		await clearAllData();
		experiments = [];
		config = { ...DEFAULT_CONFIG };
		setSelectedExp(null);
		inferences = [];
		currentExpDbId = null;
	}

	async function handleExport() {
		const blob = await exportCsvZip();
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'autoresearch-experiments.zip';
		a.click();
		URL.revokeObjectURL(url);
	}

	let paramCount = $derived(estimateParams(config));
	let paramCapExceeded = $derived(paramCount > maxParams);

	let currentInference = $derived(inferences.length > 0 ? inferences[inferenceIdx] : null);
	let selectedExp = $derived(selectedExpId ? experiments.find(e => e.id === selectedExpId) ?? null : null);

	/** Previous experiment's config, for showing diff. */
	let prevExpConfig = $derived.by(() => {
		if (!selectedExp) return null;
		const idx = experiments.findIndex(e => e.id === selectedExp!.id);
		if (idx <= 0) return null;
		return experiments[idx - 1].config;
	});
	/** True when viewing an old experiment's config (inputs should be locked). */
	let viewingExisting = $derived(!!selectedExpId && !running);

	function forkFromSelected() {
		if (!selectedExp) return;
		const baseName = selectedExp.name;
		experimentName = `${baseName} (fork)`;
		experimentDesc = '';
		// config is already set from selectExperiment — just unlock
		setSelectedExp(null);
	}
</script>

<svelte:head>
	<title>autoresearch-webgpu</title>
</svelte:head>

<main class="px-6 py-6 max-w-6xl mx-auto space-y-5">
	{#if gpuStatus === null}
		<p class="text-gray-400 font-mono text-sm">initializing webgpu...</p>
	{:else if !gpuStatus.ok}
		<div class="rounded border border-red-800 bg-red-950 p-4 font-mono text-sm text-red-400">
			{gpuStatus.reason}
		</div>
	{:else}
		<div class="max-w-[50%]">
			<h1 class="text-lg font-mono text-gray-200">autoresearch-webgpu</h1>
			<p class="text-xs font-mono text-gray-500">
				Based on Andrej Karpathy's <a href="https://github.com/karpathy/autoresearch" class="underline hover:text-gray-300">autoresearch</a> and built on Eric Zhang's <a href="https://github.com/ekzhang/jax-js" class="underline hover:text-gray-300">jax-js</a>. Built by <a href="https://lucasgelfond.online" class="underline hover:text-gray-300">Lucas Gelfond</a>. Source <a href="https://github.com/lucasgelfond/autoresearch-webgpu" class="underline hover:text-gray-300">here</a>.
			</p>
		</div>
		<div class="flex items-center gap-3">
			<div class="flex rounded border border-gray-700 text-xs font-mono overflow-hidden">
				<button
					class="px-2.5 py-0.5 {mode === 'manual' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}"
					onclick={() => (mode = 'manual')}
					disabled={running}
				>
					manual
				</button>
				<button
					class="px-2.5 py-0.5 {mode === 'research' ? 'bg-blue-700 text-white' : 'text-gray-400 hover:text-white'}"
					onclick={() => (mode = 'research')}
					disabled={running}
				>
					auto
				</button>
			</div>
			<button
				onclick={() => (showConstraints = true)}
				class="text-gray-400 hover:text-white p-1"
				title="set constraints"
			>
				<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
					<path fill-rule="evenodd" d="M7.84 1.804A1 1 0 0 1 8.82 1h2.36a1 1 0 0 1 .98.804l.331 1.652a6.993 6.993 0 0 1 1.929 1.115l1.598-.54a1 1 0 0 1 1.186.447l1.18 2.044a1 1 0 0 1-.205 1.251l-1.267 1.113a7.047 7.047 0 0 1 0 2.228l1.267 1.113a1 1 0 0 1 .206 1.25l-1.18 2.045a1 1 0 0 1-1.187.447l-1.598-.54a6.993 6.993 0 0 1-1.929 1.115l-.33 1.652a1 1 0 0 1-.98.804H8.82a1 1 0 0 1-.98-.804l-.331-1.652a6.993 6.993 0 0 1-1.929-1.115l-1.598.54a1 1 0 0 1-1.186-.447l-1.18-2.044a1 1 0 0 1 .205-1.251l1.267-1.114a7.05 7.05 0 0 1 0-2.227L1.821 7.773a1 1 0 0 1-.206-1.25l1.18-2.045a1 1 0 0 1 1.187-.447l1.598.54A6.992 6.992 0 0 1 7.51 3.456l.33-1.652ZM10 13a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" clip-rule="evenodd" />
				</svg>
			</button>
		</div>

		<div class="grid grid-cols-[240px_1fr_240px] gap-4 h-[calc(100vh-13rem)]">
			<!-- Left: config + controls -->
			<div class="space-y-2 overflow-y-auto h-full">
				<div class="rounded border border-gray-800 p-3">
					<h2 class="text-xs font-mono text-gray-400 mb-2">config</h2>
					<ConfigEditor bind:config disabled={running} {constraints} />
				</div>

				{#if paramCapExceeded}
					<div class="rounded border border-yellow-800 bg-yellow-950/50 px-3 py-2 font-mono text-xs text-yellow-400">
						{(paramCount / 1e6).toFixed(1)}M params exceeds {(maxParams / 1e6).toFixed(0)}M cap
					</div>
				{/if}

				{#if mode === 'manual'}
					<input
						type="text"
						bind:value={experimentName}
						placeholder="experiment name..."
						disabled={running}
						class="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
					/>
					<textarea
						bind:value={experimentDesc}
						placeholder="description / hypothesis..."
						disabled={running}
						rows={2}
						class="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 font-mono text-[11px] text-gray-200 placeholder-gray-500 disabled:opacity-40 resize-none"
					></textarea>
					{#if running}
						<button
							onclick={stopCurrentRun}
							class="w-full rounded bg-red-600 hover:bg-red-500 px-3 py-1.5 font-mono text-xs transition-colors"
						>
							stop training
						</button>
					{:else}
						<button
							onclick={startManualTraining}
							disabled={status === 'loading data...' || paramCapExceeded}
							class="w-full rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-3 py-1.5 font-mono text-xs transition-colors"
						>
							start training
						</button>
					{/if}
				{:else}
					{#if !running}
						<button
							onclick={startResearch}
							disabled={status === 'loading data...'}
							class="w-full rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-3 py-1.5 font-mono text-xs transition-colors"
						>
							start research
						</button>
					{:else}
						<button
							onclick={stopCurrentRun}
							class="w-full rounded bg-red-600 hover:bg-red-500 px-3 py-1.5 font-mono text-xs transition-colors"
						>
							stop
						</button>
					{/if}
				{/if}
			</div>

			<!-- Center: chart + status + inference -->
			<div class="flex flex-col h-full overflow-hidden">
				<div class="rounded border border-gray-800 p-3 space-y-2 flex-1 overflow-y-auto">
					{#if selectedExp}
						<div class="flex items-center gap-2 font-mono text-xs">
							<span class="px-1 py-0.5 rounded text-[10px] {selectedExp.source === 'auto' ? 'bg-blue-900/50 text-blue-300' : 'bg-gray-700 text-gray-300'}">
								{selectedExp.source === 'auto' ? 'auto' : 'manual'}
							</span>
							<span class="text-gray-200">{selectedExp.name}</span>
							<span class="text-gray-500 tabular-nums ml-auto">{selectedExp.valBpb.toFixed(4)} bpb</span>
						</div>
						{#if prevExpConfig}
							<ConfigDiff before={prevExpConfig} after={selectedExp.config} />
						{/if}
						{#if selectedExp.reasoning && selectedExp.reasoning !== selectedExp.name}
							<p class="text-[11px] text-gray-400 font-mono line-clamp-2" title={selectedExp.reasoning}>{selectedExp.reasoning}</p>
						{/if}
					{:else if running}
						<div class="flex items-center gap-2 font-mono text-xs">
							<span class="px-1 py-0.5 rounded text-[10px] bg-blue-900/50 text-blue-300 animate-pulse">
								{waitingForRecommendation ? 'thinking' : 'running'}
							</span>
							<span class="text-gray-200">
								{waitingForRecommendation ? 'Claude generating experiment ideas...' : currentRunName || 'training...'}
							</span>
						</div>
						{#if currentReasoning && !waitingForRecommendation}
							<p class="text-[11px] text-gray-400 font-mono line-clamp-2" title={currentReasoning}>{currentReasoning}</p>
						{/if}
						<p class="text-[11px] text-gray-500 font-mono">{status}</p>
					{:else}
						<h2 class="text-xs font-mono text-gray-400">loss</h2>
					{/if}
					<div class="h-44">
						<LossChart data={lossData} pastRuns={pastLossRuns} />
					</div>

					<!-- Inference -->
					<div class="border-t border-gray-800 pt-2 space-y-1.5">
						<h2 class="text-xs font-mono text-gray-400">inference</h2>
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
								title={!selectedExpId ? 'train a model to run inference' : ''}
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
						{/if}
					</div>
				</div>

				</div>

			<!-- Right: leaderboard / research log -->
			<div class="flex flex-col h-full overflow-hidden">
				<div class="rounded border border-gray-800 p-3 flex flex-col flex-1 min-h-0">
					<div class="flex items-center justify-between mb-2 shrink-0">
						<div class="flex items-center gap-1 text-xs font-mono">
							<button
								class="{listMode === 'leaderboard' ? 'text-gray-200' : 'text-gray-500 hover:text-gray-300'}"
								onclick={() => setListMode('leaderboard')}
							>leaderboard</button>
							<span class="text-gray-600">/</span>
							<button
								class="{listMode === 'current' ? 'text-gray-200' : 'text-gray-500 hover:text-gray-300'}"
								onclick={() => setListMode('current')}
							>history</button>
						</div>
						<span class="text-gray-500 text-[10px] font-mono">{experiments.length}</span>
					</div>
					<div class="flex-1 min-h-0 overflow-y-auto">
						<Leaderboard experiments={allExperiments} onSelect={selectExperiment} selected={selectedExpId ? experiments.find(e => e.id === selectedExpId) ?? null : null} sortByLoss={listMode === 'leaderboard'} />
					</div>
					{#if experiments.length > 0}
						<div class="flex gap-3 mt-2 pt-2 border-t border-gray-800 shrink-0">
							<button onclick={handleExport} class="text-gray-500 hover:text-gray-300 text-xs font-mono">
								export
							</button>
							<button onclick={handleClear} disabled={running} class="text-gray-500 hover:text-red-400 text-xs font-mono">
								clear
							</button>
						</div>
					{/if}
				</div>
			</div>
		</div>
	{/if}

	{#if showConstraints}
		<ConstraintsModal bind:constraints onClose={() => (showConstraints = false)} />
	{/if}

	{#if showClearModal}
		<div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
			<div class="bg-gray-900 border border-gray-700 rounded-lg p-6 max-w-sm space-y-4 font-mono">
				<h3 class="text-sm text-gray-200">clear all data?</h3>
				<p class="text-xs text-gray-400">this will delete all experiments, loss curves, inferences, and saved weights. this cannot be undone.</p>
				<div class="flex gap-2 justify-end">
					<button
						onclick={() => (showClearModal = false)}
						class="px-3 py-1.5 rounded bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm"
					>cancel</button>
					<button
						onclick={confirmClear}
						class="px-3 py-1.5 rounded bg-red-600 text-white hover:bg-red-500 text-sm"
					>clear everything</button>
				</div>
			</div>
		</div>
	{/if}
</main>
