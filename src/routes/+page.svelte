<script lang="ts">
	import { onMount } from 'svelte';
	import { initWebGPU, type WebGPUStatus } from '$lib/webgpu';
	import { DataLoader } from '$lib/data/loader';
	import { executeTrainCode, type RunResult } from '$lib/research/sandbox';
	import type { StepMetrics, ForwardFn, Params } from '$lib/prepare';
	import { sampleText } from '$lib/sample';
	import { ResearchController } from '$lib/research/controller';
	import type { ExperimentRecord } from '$lib/research/prompt';
	import { BASELINE_CODE } from '$lib/research/baseline';
	import {
		getDb, insertExperiment, insertInference, insertLossCurve,
		getBestExperiment, getInferencesForExperiment, clearAllData, deleteExperiments,
		exportCsvZip, importCsvZip, updateWeightsPath, getAllExperimentRecords, type InferenceRow
	} from '$lib/db';
	import { saveWeights, loadWeights } from '$lib/weights';
	import LossChart from '$lib/components/LossChart.svelte';
	import CodeEditor from '$lib/components/CodeEditor.svelte';
	import Leaderboard from '$lib/components/Leaderboard.svelte';
	import EndpointManager from '$lib/components/EndpointManager.svelte';
	import { petname } from '$lib/petname';
	import { isConfiguredProfile, type ResearchEndpointProfile } from '$lib/research/providers';
	import {
		loadActiveResearchProfileId,
		loadResearchProfiles,
		saveActiveResearchProfileId,
		saveResearchProfiles
	} from '$lib/research/profile-store';

	let gpuStatus = $state<WebGPUStatus | null>(null);
	let code = $state(BASELINE_CODE);
	let running = $state(false);
	let lossData = $state<{ step: number; loss: number }[]>([]);
	let status = $state<string>('initializing');
	let sampling = $state(false);
	let mode = $state<'manual' | 'research'>('research');
	let experiments = $state<ExperimentRecord[]>([]);
	let currentReasoning = $state('');
	let experimentName = $state(petname());
	let listMode = $state<'leaderboard' | 'current'>('leaderboard');

	// Inference state — we need to store forward fn + params for sampling
	let prompt = $state('');
	let temperature = $state(0.8);
	let selectedExpId = $state<number | null>(null);
	let inferences = $state<InferenceRow[]>([]);
	let inferenceIdx = $state(0);
	let streamingOutput = $state('');
	let currentRunName = $state('');
	let trainAbort: AbortController | null = null;
	let inProgressExp = $state<ExperimentRecord | null>(null);
	let waitingForRecommendation = $state(false);
	let researchProfiles = $state<ResearchEndpointProfile[]>([]);
	let selectedResearchProfileId = $state<string | null>(null);
	let importing = $state(false);
	let importInput = $state<HTMLInputElement | null>(null);
	let selectionMode = $state(false);
	let selectedBatchIds = $state<number[]>([]);
	let workspaceTab = $state<'chart' | 'code' | 'inference'>('chart');
	let backendCollapsed = $state(false);
	let leaderboardSort = $state<'bpb' | 'newest' | 'oldest' | 'steps' | 'name'>('bpb');
	let chartSeriesMode = $state<'all' | 'focus'>('all');
	let chartScaleMode = $state<'fit' | 'trim' | 'manual'>('fit');
	let chartYMinInput = $state('');
	let chartYMaxInput = $state('');

	// In-memory loaded model state for inference
	type LoadedModel = { forward: ForwardFn; params: Params; vocabSize: number; seqLen: number; expId: number };
	let loadedModel = $state<LoadedModel | null>(null);

	let allExperiments = $derived(
		inProgressExp ? [...experiments, inProgressExp] : experiments
	);
	let selectedBatchExperiments = $derived(
		experiments.filter((exp) => selectedBatchIds.includes(exp.id))
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
	let selectedResearchProfile = $derived(
		selectedResearchProfileId
			? researchProfiles.find((profile) => profile.id === selectedResearchProfileId) ?? null
			: null
	);
	let readyResearchProfile = $derived(
		isConfiguredProfile(selectedResearchProfile) ? selectedResearchProfile : null
	);
	let configuredProfileCount = $derived(
		researchProfiles.filter((profile) => isConfiguredProfile(profile)).length
	);
	let backendSummary = $derived.by(() => {
		if (readyResearchProfile) {
			return `${readyResearchProfile.name || readyResearchProfile.model} ready`;
		}
		if (configuredProfileCount > 0) {
			return `${configuredProfileCount} configured profiles`;
		}
		return 'no configured backend';
	});
	let activeResearchLabel = $derived.by(() => {
		if (!readyResearchProfile) return 'research backend';
		const name = readyResearchProfile.name.trim();
		if (name) return name;
		const model = readyResearchProfile.model.trim();
		if (model) return model;
		return readyResearchProfile.provider === 'openai' ? 'OpenAI-compatible backend' : 'Anthropic-compatible backend';
	});
	let canFocusChart = $derived(Boolean(selectedExpId) || lossData.length >= 2);
	let chartYMin = $derived.by(() => {
		const trimmed = String(chartYMinInput ?? '').trim();
		if (trimmed === '') return null;
		const value = Number(trimmed);
		return Number.isFinite(value) ? value : null;
	});
	let chartYMax = $derived.by(() => {
		const trimmed = String(chartYMaxInput ?? '').trim();
		if (trimmed === '') return null;
		const value = Number(trimmed);
		return Number.isFinite(value) ? value : null;
	});
	let chartManualRangeValid = $derived(
		chartScaleMode !== 'manual' ||
		(chartYMin != null && chartYMax != null && chartYMax > chartYMin)
	);

	onMount(async () => {
		try {
			status = 'initializing';
			await getDb();
			researchProfiles = loadResearchProfiles();
			selectedResearchProfileId = loadActiveResearchProfileId(researchProfiles);
			backendCollapsed = researchProfiles.some((profile) => isConfiguredProfile(profile));
			gpuStatus = await initWebGPU();
			if (!gpuStatus.ok) { status = 'error'; return; }
			status = 'loading data...';
			[trainLoader, valLoader] = await Promise.all([
				DataLoader.fetch('/data/train.bin'),
				DataLoader.fetch('/data/val.bin')
			]);
			await loadFromDb();
			const params = new URL(window.location.href).searchParams;
			const expParam = params.get('exp');
			if (expParam) {
				const id = Number(expParam);
				const exp = experiments.find(e => e.id === id);
				if (exp) selectExperiment(exp);
			}
			status = 'ready';
		} catch (e) {
			console.error('Init failed:', e);
			status = 'error';
		}
	});

	$effect(() => {
		if (typeof window === 'undefined' || researchProfiles.length === 0) return;
		saveResearchProfiles(researchProfiles);
		saveActiveResearchProfileId(selectedResearchProfileId);
	});

	$effect(() => {
		if (chartSeriesMode === 'focus' && !canFocusChart) {
			chartSeriesMode = 'all';
		}
	});

	async function loadFromDb() {
		experiments = await getAllExperimentRecords();
		const best = [...experiments]
			.filter((exp) => !exp.error && Number.isFinite(exp.valBpb))
			.sort((a, b) => a.valBpb - b.valBpb || b.id - a.id)[0];
		if (best) {
			code = best.code;
		}
	}

	function makeBenchmarkGroupLabel() {
		return new Date().toISOString().replace('T', ' ').slice(0, 19);
	}

	function clearBatchSelection() {
		selectedBatchIds = [];
	}

	function toggleBatchSelection(expId: number) {
		selectedBatchIds = selectedBatchIds.includes(expId)
			? selectedBatchIds.filter((id) => id !== expId)
			: [...selectedBatchIds, expId];
	}

	type LocalRunRequest = {
		name: string;
		source: 'manual' | 'auto';
		code: string;
		reasoning: string;
		rerunOf?: number | null;
		benchmarkGroup?: string | null;
		saveWeights?: boolean;
		generateSample?: boolean;
		timeoutMs?: number;
	};

	type LocalRunOutcome = {
		record: ExperimentRecord;
		result: RunResult;
		dbId: number;
	};

	async function runLocalExperiment(request: LocalRunRequest): Promise<LocalRunOutcome> {
		if (!trainLoader || !valLoader) {
			throw new Error('training data is not ready');
		}

		lossData = [];
		inferences = [];
		inferenceIdx = 0;
		currentRunName = request.name;
		currentReasoning = request.reasoning;
		status = request.rerunOf
			? `rerunning #${request.rerunOf}...`
			: 'training...';
		trainAbort = new AbortController();

		inProgressExp = {
			id: -1,
			name: request.name,
			source: request.source,
			code: request.code,
			valBpb: Infinity,
			elapsed: 0,
			totalSteps: 0,
			reasoning: request.reasoning,
			kept: false,
			rerunOf: request.rerunOf ?? null,
			benchmarkGroup: request.benchmarkGroup ?? null
		};

		const result = await executeTrainCode(request.code, trainLoader, valLoader, 30, {
			signal: trainAbort.signal,
			timeoutMs: request.timeoutMs,
			onStep(m: StepMetrics) {
				lossData = [...lossData, { step: m.step, loss: m.loss }];
				status = `step ${m.step} | loss ${m.loss.toFixed(4)} | ${(m.elapsed / 1000).toFixed(1)}s`;
				if (inProgressExp) {
					inProgressExp = {
						...inProgressExp,
						valBpb: m.loss,
						totalSteps: m.step,
						elapsed: m.elapsed
					};
				}
			}
		});
		trainAbort = null;
		inProgressExp = null;

		const kept = experiments.length === 0 || result.valBpb < Math.min(...experiments.map((e) => e.valBpb));
		const dbId = await insertExperiment({
			name: request.name,
			source: request.source,
			code: request.code,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning: request.reasoning,
			kept,
			lossCurve: lossData,
			error: result.error,
			rerunOf: request.rerunOf ?? null,
			benchmarkGroup: request.benchmarkGroup ?? null
		});

		await insertLossCurve(dbId, lossData);

		if (!result.error) {
			loadedModel = {
				forward: result.forward,
				params: result.params,
				vocabSize: result.vocabSize,
				seqLen: result.seqLen,
				expId: dbId
			};
		}

		if (request.saveWeights && result.params && Object.keys(result.params).length > 0) {
			(async () => {
				try {
					const weightsPath = await saveWeights(dbId, result.params);
					await updateWeightsPath(dbId, weightsPath);
				} catch (e) {
					console.error('Failed to save weights:', e);
				}

				if (request.generateSample && loadedModel?.forward && loadedModel.expId === dbId) {
					try {
						const output = await sampleText(
							result.params,
							loadedModel.forward,
							result.vocabSize,
							result.seqLen,
							'',
							200,
							0.8
						);
						await insertInference({ experimentId: dbId, prompt: '', output, temperature: 0.8 });
						if (selectedExpId === dbId) {
							inferences = await getInferencesForExperiment(dbId);
							inferenceIdx = 0;
						}
					} catch (e) {
						console.error('Failed to generate sample:', e);
					}
				}
			})();
		}

		return {
			dbId,
			result,
			record: {
				id: dbId,
				name: request.name,
				source: request.source,
				code: request.code,
				valBpb: result.valBpb,
				elapsed: result.elapsed,
				totalSteps: result.totalSteps,
				reasoning: request.reasoning,
				kept,
				error: result.error,
				lossCurve: lossData,
				rerunOf: request.rerunOf ?? null,
				benchmarkGroup: request.benchmarkGroup ?? null
			}
		};
	}

	async function startManualTraining() {
		if (!trainLoader || !valLoader || running) return;

		running = true;
		setListMode('current');

		try {
			const runName = experimentName || petname();
			const outcome = await runLocalExperiment({
				name: runName,
				source: 'manual',
				code,
				reasoning: 'Manual run',
				saveWeights: true,
				generateSample: true
			});
			await loadFromDb();
			selectExperimentById(outcome.dbId);
			status = outcome.result.error
				? `error: ${outcome.result.error}`
				: `done — val_bpb: ${outcome.result.valBpb.toFixed(4)} | ${outcome.result.totalSteps} steps`;
			experimentName = petname();
		} finally {
			running = false;
		}
	}

	async function rerunExperiments(targets: ExperimentRecord[]) {
		if (!trainLoader || !valLoader || running || targets.length === 0) return;

		running = true;
		setListMode('current');
		const benchmarkGroup = makeBenchmarkGroupLabel();
		const saveWeights = targets.length === 1;
		let completed = 0;
		let failures = 0;
		let lastDbId: number | null = null;

		try {
			for (const exp of targets) {
				const outcome = await runLocalExperiment({
					name: exp.name,
					source: exp.source,
					code: exp.code,
					reasoning: `Benchmark rerun of #${exp.id}${exp.reasoning ? ` — ${exp.reasoning}` : ''}`,
					rerunOf: exp.id,
					benchmarkGroup,
					saveWeights,
					generateSample: false,
					timeoutMs: 180000
				});
				completed += 1;
				if (outcome.result.error) failures += 1;
				lastDbId = outcome.dbId;
				await loadFromDb();

				if (outcome.result.error?.toLowerCase().includes('aborted')) {
					status = `stopped after ${completed}/${targets.length} reruns in ${benchmarkGroup}`;
					return;
				}
			}

			if (lastDbId != null) {
				selectExperimentById(lastDbId);
			}
			status = failures > 0
				? `reran ${completed} experiments in baseline ${benchmarkGroup} (${failures} failed)`
				: `reran ${completed} experiment${completed === 1 ? '' : 's'} in baseline ${benchmarkGroup}`;
		} finally {
			running = false;
			clearBatchSelection();
			selectionMode = false;
		}
	}

	async function rerunSelectedExperiment() {
		if (!selectedExp) return;
		await rerunExperiments([selectedExp]);
	}

	async function rerunSelectedBatch() {
		if (selectedBatchExperiments.length === 0) return;
		await rerunExperiments(selectedBatchExperiments);
	}

	async function rerunAllExperiments() {
		const originals = experiments.filter((exp) => !exp.rerunOf);
		if (originals.length === 0) return;
		await rerunExperiments(originals);
	}

	async function deleteExperimentRecords(ids: number[], scopeLabel: string) {
		if (ids.length === 0 || running || importing) return;

		const confirmed = window.confirm(
			`Delete ${scopeLabel}? This removes the experiment rows, their loss curves, inferences, and any saved weights for those runs.`
		);
		if (!confirmed) return;

		const selectedIdBeforeDelete = selectedExpId;
		const deletedCount = await deleteExperiments(ids);
		await loadFromDb();

		selectedBatchIds = selectedBatchIds.filter((id) => !ids.includes(id));
		if (selectedIdBeforeDelete != null && ids.includes(selectedIdBeforeDelete)) {
			setSelectedExp(null);
			code = experiments.length > 0 ? code : BASELINE_CODE;
		}

		status = `deleted ${deletedCount} experiment${deletedCount === 1 ? '' : 's'}`;
	}

	async function deleteSelectedExperiment() {
		if (!selectedExp) return;
		await deleteExperimentRecords([selectedExp.id], `experiment #${selectedExp.id}`);
	}

	async function deleteSelectedBatch() {
		if (selectedBatchIds.length === 0) return;
		await deleteExperimentRecords(selectedBatchIds, `${selectedBatchIds.length} selected experiments`);
	}

	async function startResearch() {
		if (!trainLoader || !valLoader || running) return;
		if (!readyResearchProfile) {
			status = 'configure a research backend before starting';
			return;
		}

		running = true;
		waitingForRecommendation = true;
		lossData = [];
		status = 'starting...';
		controller = new ResearchController();
		controller.profile = readyResearchProfile;
		setListMode('current');

		controller.history = [...experiments];
		const bestExperiment = [...experiments]
			.filter((exp) => !exp.error && Number.isFinite(exp.valBpb))
			.sort((a, b) => {
				const aTime = a.createdAt ? Date.parse(a.createdAt) : NaN;
				const bTime = b.createdAt ? Date.parse(b.createdAt) : NaN;
				return (
					a.valBpb - b.valBpb ||
					(Number.isFinite(bTime) ? bTime : b.id) - (Number.isFinite(aTime) ? aTime : a.id) ||
					b.id - a.id
				);
			})[0];
		if (bestExperiment) {
			controller.bestCode = bestExperiment.code;
			controller.bestBpb = bestExperiment.valBpb;
		} else {
			const best = await getBestExperiment();
			if (best) {
				controller.bestCode = best.code;
				controller.bestBpb = best.val_bpb;
			}
		}

		await controller.run(trainLoader, valLoader, {
			onExperimentStart(expCode, reasoning) {
				waitingForRecommendation = false;
				lossData = [];
				code = expCode;
				currentReasoning = reasoning;
				currentRunName = petname();
				status = `experiment: ${reasoning}`;
				if (listMode === 'current') setSelectedExp(null);
				inProgressExp = {
					id: -1, name: currentRunName, source: 'auto', code: expCode,
					valBpb: Infinity, elapsed: 0, totalSteps: 0, reasoning, kept: false,
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
					code = controller.bestCode;
				}
				status = `#${record.id} ${record.kept ? 'KEPT' : 'discarded'} — bpb ${record.valBpb.toFixed(4)}`;
			},
			onCodeStream(streamedCode) {
				code = streamedCode;
			},
			onReasoningStream(streamedReasoning) {
				currentReasoning = streamedReasoning;
				status = `thinking: ${streamedReasoning}`;
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
		inferences = [];
		inferenceIdx = 0;
		streamingOutput = '';
		sampling = false;
		const url = new URL(window.location.href);
		if (id != null) url.searchParams.set('exp', String(id));
		else url.searchParams.delete('exp');
		history.replaceState(null, '', url);
		if (id != null) {
			getInferencesForExperiment(id).then(rows => {
				if (selectedExpId === id) inferences = rows;
			});
		}
	}

	function setListMode(m: 'leaderboard' | 'current') {
		listMode = m;
		leaderboardSort = m === 'leaderboard' ? 'bpb' : 'newest';
		if (m === 'current') setSelectedExp(null);
	}

	function selectExperimentById(id: number) {
		setSelectedExp(id);
	}

	function selectExperiment(exp: ExperimentRecord) {
		code = exp.code;
		selectExperimentById(exp.id);
	}

	async function loadModelForExperiment(expId: number): Promise<boolean> {
		const exp = experiments.find(e => e.id === expId);
		if (!exp) return false;

		// If we already have this model loaded, skip
		if (loadedModel && loadedModel.expId === expId) return true;

		// Load saved weights
		const savedParams = await loadWeights(expId);
		if (!savedParams) return false;

		// Re-execute the code with trainSeconds=0 to get the forward function
		if (!trainLoader || !valLoader) return false;
		const result = await executeTrainCode(exp.code, trainLoader, valLoader, 0, {
			signal: new AbortController().signal,
			onStep() {},
		});

		if (!result.forward || result.error) return false;

		// Use the saved weights (not the freshly-initialized ones)
		loadedModel = {
			forward: result.forward,
			params: savedParams,
			vocabSize: result.vocabSize,
			seqLen: result.seqLen,
			expId,
		};
		return true;
	}

	async function generateSample() {
		if (sampling || !selectedExpId) return;
		sampling = true;
		try {
			if (!loadedModel || loadedModel.expId !== selectedExpId) {
				status = 'loading model...';
				const loaded = await loadModelForExperiment(selectedExpId);
				if (!loaded) {
					console.error('Could not load model for experiment', selectedExpId);
					sampling = false;
					status = 'ready';
					return;
				}
				status = 'ready';
			}
			streamingOutput = '';
			const output = await sampleText(loadedModel!.params, loadedModel!.forward, loadedModel!.vocabSize, loadedModel!.seqLen, prompt, 200, temperature, (text) => {
				streamingOutput = text;
			});
			await insertInference({ experimentId: selectedExpId, prompt, output, temperature });
			streamingOutput = '';
			inferences = await getInferencesForExperiment(selectedExpId);
			inferenceIdx = 0;
		} catch (e) {
			console.error('Inference failed:', e);
		}
		sampling = false;
	}

	let showClearModal = $state(false);

	function handleClear() { showClearModal = true; }

	async function confirmClear() {
		showClearModal = false;
		await clearAllData();
		experiments = [];
		code = BASELINE_CODE;
		setSelectedExp(null);
		clearBatchSelection();
		selectionMode = false;
		inferences = [];
		loadedModel = null;
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

	function handleImportClick() {
		if (running || importing) return;
		importInput?.click();
	}

	async function handleImportFile(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		if (!file) return;

		importing = true;
		status = 'importing experiments...';

		try {
			const summary = await importCsvZip(file);
			await loadFromDb();
			status = `imported ${summary.addedExperiments} experiments, ${summary.addedLossSteps} loss steps, ${summary.addedInferences} inferences`;
		} catch (e) {
			const message = e instanceof Error ? e.message : String(e);
			status = `import failed: ${message}`;
		} finally {
			importing = false;
			input.value = '';
		}
	}

	let currentInference = $derived(inferences.length > 0 ? inferences[inferenceIdx] : null);
	let selectedExp = $derived(selectedExpId ? experiments.find(e => e.id === selectedExpId) ?? null : null);
	let hasAnyModel = $derived(experiments.length > 0 || running);
	let isFirstLoad = $derived(experiments.length === 0 && !running && status !== 'error');
</script>

<svelte:head>
	<title>autoresearch, in the browser!</title>
	<meta name="description" content="Train small language models in your browser using WebGPU. Claude writes the training code, runs experiments, and iterates — a model training another model!" />
	<meta property="og:title" content="autoresearch, in the browser!" />
	<meta property="og:description" content="Train small language models in your browser using WebGPU. Claude writes the training code, runs experiments, and iterates." />
	<meta property="og:image" content="https://autoresearch.lucasgelfond.online/demo.gif" />
	<meta property="og:type" content="website" />
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content="autoresearch, in the browser!" />
	<meta name="twitter:description" content="Train small language models in your browser using WebGPU. Claude writes the training code and iterates." />
	<meta name="twitter:image" content="https://autoresearch.lucasgelfond.online/demo.gif" />
</svelte:head>

<main class="px-6 py-12 max-w-6xl mx-auto space-y-5">
	<!-- Mobile warning -->
	<div class="md:hidden rounded border-2 border-red-600 bg-red-950 p-4 font-mono text-sm text-red-300 text-center leading-relaxed">
		<span class="font-bold text-red-400">WARNING:</span> mobile devices often crash with memory intensive web activities. best results on a proper computer!
	</div>

	{#if gpuStatus && !gpuStatus.ok}
		<div class="rounded border border-red-800 bg-red-950 p-4 font-mono text-sm text-red-400">
			{gpuStatus.reason}
		</div>
	{:else}
		<div class="max-w-xl space-y-3">
			<h1 class="text-lg font-mono font-bold text-white">autoresearch, in the browser!</h1>
			<p class="text-xs font-mono text-gray-300 leading-relaxed">
				Train a small language model (all on device, with WebGPU!) by generating training code, running experiments, and iterating on progress.
			</p>
			<p class="text-[10px] font-mono text-gray-500 leading-relaxed">
				Based on Andrej Karpathy's <a href="https://github.com/karpathy/autoresearch" class="underline hover:text-gray-200">autoresearch</a> and built on <a href="https://www.ekzhang.com/" class="underline hover:text-gray-200">Eric Zhang</a>'s <a href="https://github.com/ekzhang/jax-js" class="underline hover:text-gray-200">jax-js</a>. Built by <a href="https://lucasgelfond.online" class="underline hover:text-gray-200">Lucas Gelfond</a>. Source <a href="https://github.com/lucasgelfond/autoresearch-webgpu" class="underline hover:text-gray-200">here</a>.
			</p>
		</div>
		{#if status === 'error'}
			<div class="rounded border border-red-800 bg-red-950 p-4 font-mono text-sm text-red-400 mt-4">
				something went wrong during initialization. check the console for details.
			</div>
		{:else if isFirstLoad}
		<div class="space-y-4">
			<EndpointManager bind:profiles={researchProfiles} bind:selectedId={selectedResearchProfileId} disabled={running || importing} />
			<div class="rounded border border-gray-800 bg-gray-950/60 p-6 flex flex-col items-center justify-center space-y-4 mx-auto" style="min-height: calc(100vh - 18rem);">
				<input bind:this={importInput} type="file" accept=".zip,application/zip" class="hidden" onchange={handleImportFile} />
				{#if status === 'initializing' || status === 'loading data...'}
					<p class="text-sm font-mono text-gray-500">{status}</p>
				{:else}
					<div class="flex flex-col items-center gap-3">
						<button
							onclick={() => { mode = 'research'; startResearch(); }}
							disabled={running || importing}
							class="rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-8 py-4 font-mono text-sm text-white transition-colors"
						>
							start research
						</button>
						<button
							onclick={handleImportClick}
							disabled={running || importing}
							class="rounded border border-gray-700 px-5 py-2 font-mono text-xs text-gray-300 hover:border-gray-500 hover:text-white disabled:opacity-40 transition-colors"
						>
							{importing ? 'importing...' : 'import results zip'}
						</button>
					</div>
					<p class="text-xs font-mono text-gray-500 max-w-sm text-center">
						Configure a backend to run new research, or import a prior export ZIP to restore experiment history from another machine.
					</p>
					{#if status !== 'ready'}
						<p class="text-sm font-mono text-amber-300 text-center">{status}</p>
					{/if}
				{/if}
			</div>
		</div>
		{:else}
		<div class="space-y-4">
			<div class="rounded border border-gray-800 bg-gray-950/60">
				<button
					type="button"
					class="w-full flex items-center justify-between gap-3 px-3 py-2 text-left"
					onclick={() => (backendCollapsed = !backendCollapsed)}
				>
					<div>
						<h2 class="text-xs font-mono text-gray-300">research backends</h2>
						<p class="mt-1 text-[10px] font-mono text-gray-500">{backendSummary}</p>
					</div>
					<span class="text-xs font-mono text-gray-500">{backendCollapsed ? 'show' : 'hide'}</span>
				</button>
				{#if !backendCollapsed}
					<div class="border-t border-gray-800 p-3">
						<EndpointManager bind:profiles={researchProfiles} bind:selectedId={selectedResearchProfileId} disabled={running || importing} showHeader={false} />
					</div>
				{/if}
			</div>

			<div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_280px] gap-4">
				<div class="flex flex-col gap-4 min-w-0">
					<div class="rounded border border-gray-800 bg-gray-950/60 p-3 space-y-3">
						<div class="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
							<div class="flex items-center gap-3">
								<div class="flex rounded border border-gray-700 text-xs font-mono overflow-hidden">
									<button
										class="px-2.5 py-0.5 {mode === 'manual' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'}"
										onclick={() => (mode = 'manual')}
										disabled={running}
									>manual</button>
									<button
										class="px-2.5 py-0.5 {mode === 'research' ? 'bg-blue-700 text-white' : 'text-gray-400 hover:text-white'}"
										onclick={() => (mode = 'research')}
										disabled={running}
									>auto</button>
								</div>
								<p class="text-[11px] font-mono text-gray-500">{status}</p>
							</div>
							<div class="flex flex-col gap-2 md:flex-row md:items-center md:justify-end md:flex-1">
								{#if mode === 'manual'}
									<input
										type="text"
										bind:value={experimentName}
										placeholder="experiment name..."
										disabled={running}
										class="w-full md:w-56 bg-gray-800 border border-gray-700 rounded px-2 py-1 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
									/>
								{/if}
								<button
									onclick={() => (code = BASELINE_CODE)}
									disabled={running}
									class="rounded border border-gray-700 px-3 py-1.5 text-[10px] font-mono text-gray-400 hover:text-white hover:border-gray-500 disabled:opacity-30"
								>reset code</button>
								{#if running}
									<button onclick={stopCurrentRun}
										class="rounded bg-red-600 hover:bg-red-500 px-3 py-1.5 font-mono text-xs transition-colors">
										{mode === 'manual' ? 'stop training' : 'stop'}
									</button>
								{:else if mode === 'manual'}
									<button onclick={startManualTraining}
										disabled={status === 'loading data...' || importing}
										class="rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-3 py-1.5 font-mono text-xs transition-colors">
										run train.ts
									</button>
								{:else}
									<button onclick={startResearch}
										disabled={status === 'loading data...' || importing}
										class="rounded bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 px-3 py-1.5 font-mono text-xs transition-colors">
										start research
									</button>
								{/if}
							</div>
						</div>

						<div class="rounded border border-gray-800 bg-black/20 p-3 min-h-[6.5rem]">
							{#if selectedExp}
								<div class="flex items-center gap-2 font-mono text-xs">
									<span class="px-1 py-0.5 rounded text-[10px] {selectedExp.source === 'auto' ? 'bg-blue-900/50 text-blue-300' : 'bg-gray-700 text-gray-300'}">
										{selectedExp.source === 'auto' ? 'auto' : 'manual'}
									</span>
									<span class="text-gray-200">{selectedExp.name}</span>
									<span class="text-gray-500 tabular-nums ml-auto">{selectedExp.valBpb.toFixed(4)} bpb</span>
								</div>
								{#if selectedExp.rerunOf || selectedExp.benchmarkGroup}
									<p class="mt-2 text-[10px] text-amber-300 font-mono">
										{#if selectedExp.rerunOf}rerun of #{selectedExp.rerunOf}{/if}
										{#if selectedExp.rerunOf && selectedExp.benchmarkGroup} · {/if}
										{#if selectedExp.benchmarkGroup}baseline {selectedExp.benchmarkGroup}{/if}
									</p>
								{/if}
								{#if selectedExp.reasoning}
									<p class="mt-2 text-[11px] text-gray-400 font-mono line-clamp-2" title={selectedExp.reasoning}>{selectedExp.reasoning}</p>
								{/if}
								{#if selectedExp.error}
									<p class="mt-2 text-[11px] text-red-400 font-mono line-clamp-2">error: {selectedExp.error}</p>
								{/if}
								<div class="mt-3 flex items-center gap-2">
									<button
										onclick={rerunSelectedExperiment}
										disabled={running || importing}
										class="rounded border border-gray-700 px-2 py-1 font-mono text-[10px] text-gray-300 hover:border-gray-500 hover:text-white disabled:opacity-40 transition-colors"
									>
										rerun this code
									</button>
									<button
										onclick={deleteSelectedExperiment}
										disabled={running || importing}
										class="rounded border border-red-900 px-2 py-1 font-mono text-[10px] text-red-300 hover:border-red-700 hover:text-red-200 disabled:opacity-40 transition-colors"
									>
										delete this run
									</button>
								</div>
							{:else if running}
								<div class="flex items-center gap-2 font-mono text-xs">
									<span class="px-1 py-0.5 rounded text-[10px] bg-blue-900/50 text-blue-300 animate-pulse">
										{waitingForRecommendation ? 'thinking' : 'running'}
									</span>
									<span class="text-gray-200">
										{waitingForRecommendation ? `${activeResearchLabel} writing code...` : currentRunName || 'training...'}
									</span>
								</div>
								{#if currentReasoning && !waitingForRecommendation}
									<p class="mt-2 text-[11px] text-gray-400 font-mono line-clamp-2" title={currentReasoning}>{currentReasoning}</p>
								{/if}
								<p class="mt-2 text-[11px] text-gray-500 font-mono">{status}</p>
							{:else}
								<div class="space-y-1">
									<h2 class="text-xs font-mono text-gray-400">workspace</h2>
									<p class="text-[11px] font-mono text-gray-500">
										Use the tabs below to inspect the loss chart, edit the training code, or run inference on the selected experiment.
									</p>
								</div>
							{/if}
						</div>
					</div>

					<div class="rounded border border-gray-800 bg-gray-950/60 flex flex-col min-h-[34rem] xl:h-[calc(100vh-26rem)] overflow-hidden">
						<div class="border-b border-gray-800 px-3 pt-3">
							<div class="flex items-center gap-2">
								<button
									type="button"
									onclick={() => (workspaceTab = 'chart')}
									class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {workspaceTab === 'chart' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
								>chart</button>
								<button
									type="button"
									onclick={() => (workspaceTab = 'code')}
									class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {workspaceTab === 'code' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
								>code</button>
								<button
									type="button"
									onclick={() => (workspaceTab = 'inference')}
									class="rounded-t border border-b-0 px-3 py-1.5 text-xs font-mono transition-colors {workspaceTab === 'inference' ? 'border-gray-700 bg-gray-900 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'}"
								>inference</button>
							</div>
						</div>

						<div class="flex-1 min-h-0 p-3">
							{#if workspaceTab === 'chart'}
								<div class="h-full flex flex-col gap-3">
									<div class="flex flex-col gap-2 xl:flex-row xl:items-start xl:justify-between">
										<div>
											<h2 class="text-xs font-mono text-gray-400">loss</h2>
											{#if selectedExp}
												<span class="text-[10px] font-mono text-gray-500 tabular-nums">
													{selectedExp.totalSteps} steps · {(selectedExp.elapsed / 1000).toFixed(1)}s
												</span>
											{/if}
										</div>
										<div class="flex flex-wrap items-center gap-2 text-[10px] font-mono text-gray-400">
											<select
												bind:value={chartSeriesMode}
												disabled={!canFocusChart}
												class="rounded border border-gray-800 bg-black/40 px-2 py-1 disabled:opacity-40"
												title="chart series view"
											>
												<option value="all">view: all runs</option>
												<option value="focus">view: focus selected</option>
											</select>
											<select
												bind:value={chartScaleMode}
												class="rounded border border-gray-800 bg-black/40 px-2 py-1"
												title="chart y-axis scaling"
											>
												<option value="fit">y: fit all</option>
												<option value="trim">y: trim outliers</option>
												<option value="manual">y: manual range</option>
											</select>
											{#if chartScaleMode === 'manual'}
												<input
													type="number"
													step="0.01"
													bind:value={chartYMinInput}
													placeholder="y min"
													class="w-20 rounded border border-gray-800 bg-black/40 px-2 py-1 text-right tabular-nums"
													title="manual minimum y-axis value"
												/>
												<input
													type="number"
													step="0.01"
													bind:value={chartYMaxInput}
													placeholder="y max"
													class="w-20 rounded border border-gray-800 bg-black/40 px-2 py-1 text-right tabular-nums"
													title="manual maximum y-axis value"
												/>
												{#if !chartManualRangeValid}
													<span class="text-amber-300">enter y-max greater than y-min</span>
												{/if}
											{/if}
										</div>
									</div>
									<div class="flex-1 min-h-0 rounded border border-gray-800 bg-black/20 p-3">
										<LossChart
											data={lossData}
											pastRuns={pastLossRuns}
											seriesMode={chartSeriesMode}
											yScaleMode={chartScaleMode}
											yMin={chartYMin}
											yMax={chartYMax}
										/>
									</div>
								</div>
							{:else if workspaceTab === 'code'}
								<div class="h-full flex flex-col min-h-0">
									<div class="flex items-center justify-between mb-2 shrink-0">
										<h2 class="text-xs font-mono text-gray-400">train.ts</h2>
										<span class="text-[10px] font-mono text-gray-500">
											{running && mode === 'research' ? 'locked during auto runs' : 'editable'}
										</span>
									</div>
									<div class="flex-1 min-h-0">
										<CodeEditor bind:value={code} disabled={running && mode === 'research'} />
									</div>
								</div>
							{:else}
								<div class="h-full flex flex-col min-h-0 gap-2">
									<div class="flex items-center justify-between gap-3 shrink-0">
										<h2 class="text-xs font-mono text-gray-400">inference</h2>
										{#if selectedExp}
											<span class="text-[10px] font-mono text-gray-500">selected #{selectedExp.id}</span>
										{/if}
									</div>
									{#if hasAnyModel}
										<div class="flex items-center gap-2 shrink-0">
											<input
												type="text"
												bind:value={prompt}
												placeholder="prompt..."
												disabled={!selectedExpId || sampling}
												class="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 font-mono text-xs text-gray-200 placeholder-gray-500 disabled:opacity-40"
												onkeydown={(e: KeyboardEvent) => { if (e.key === 'Enter') generateSample(); }}
											/>
											<input type="number" bind:value={temperature} min={0.1} max={2} step={0.1}
												disabled={!selectedExpId || sampling}
												class="w-12 bg-gray-800 border border-gray-700 rounded px-1 py-1 text-right tabular-nums text-xs text-gray-200 font-mono disabled:opacity-40"
												title="temperature" />
											<button onclick={generateSample} disabled={!selectedExpId || sampling}
												class="rounded bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 px-3 py-1 font-mono text-xs transition-colors">
												{#if sampling}
													<svg class="animate-spin h-3.5 w-3.5 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
														<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
														<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
													</svg>
												{:else}
													go
												{/if}
											</button>
										</div>
										<div class="flex-1 min-h-0 overflow-y-auto rounded border border-gray-800 bg-black/20 p-3">
											{#if sampling && streamingOutput}
												<pre class="text-xs text-gray-300 whitespace-pre-wrap break-all font-mono leading-relaxed">{streamingOutput}<span class="animate-pulse">▌</span></pre>
											{:else if currentInference}
												{#if inferences.length > 1}
													<div class="flex items-center justify-end text-xs font-mono text-gray-500 mb-1">
														<div class="flex items-center gap-1">
															<button onclick={() => { inferenceIdx = Math.min(inferenceIdx + 1, inferences.length - 1); }}
																disabled={inferenceIdx >= inferences.length - 1} class="px-1 hover:text-gray-300 disabled:opacity-30">←</button>
															<span>{inferences.length - inferenceIdx}/{inferences.length}</span>
															<button onclick={() => { inferenceIdx = Math.max(inferenceIdx - 1, 0); }}
																disabled={inferenceIdx <= 0} class="px-1 hover:text-gray-300 disabled:opacity-30">→</button>
														</div>
													</div>
												{/if}
												<pre class="text-xs text-gray-300 whitespace-pre-wrap break-all font-mono leading-relaxed">{currentInference.output}</pre>
											{:else}
												<div class="h-full flex items-center justify-center text-center">
													<p class="text-[11px] font-mono text-gray-500 max-w-sm">
														Select an experiment with saved weights, then sample from it here.
													</p>
												</div>
											{/if}
										</div>
									{:else}
										<div class="flex-1 min-h-0 rounded border border-gray-800 bg-black/20 flex items-center justify-center text-center p-6">
											<p class="text-[11px] font-mono text-gray-500 max-w-sm">
												Train or import experiments first, then select one to use the inference workspace.
											</p>
										</div>
									{/if}
								</div>
							{/if}
						</div>
					</div>
				</div>

				<!-- Right: leaderboard -->
				<div class="flex flex-col xl:h-[calc(100vh-20rem)] xl:overflow-hidden">
				<div class="rounded border border-gray-800 p-3 flex flex-col flex-1 min-h-0">
					<div class="mb-2 shrink-0 space-y-2">
						<div class="flex items-center justify-between gap-3">
							<div class="flex items-center gap-1 text-xs font-mono">
								<button class="{listMode === 'leaderboard' ? 'text-gray-200' : 'text-gray-500 hover:text-gray-300'}"
									onclick={() => setListMode('leaderboard')}>leaderboard</button>
								<span class="text-gray-600">/</span>
								<button class="{listMode === 'current' ? 'text-gray-200' : 'text-gray-500 hover:text-gray-300'}"
									onclick={() => setListMode('current')}>history</button>
							</div>
							<div class="flex items-center gap-2 text-[10px] font-mono text-gray-500">
								<span>{experiments.length} runs</span>
								<button
									onclick={() => {
										selectionMode = !selectionMode;
										if (!selectionMode) clearBatchSelection();
									}}
									disabled={running || importing}
									class="{selectionMode ? 'text-blue-300' : 'text-gray-500 hover:text-gray-300'} disabled:opacity-40"
								>
									{selectionMode ? 'done' : 'select'}
								</button>
							</div>
						</div>
						<select
							bind:value={leaderboardSort}
							class="w-full rounded border border-gray-800 bg-black/40 px-2 py-1 text-[10px] font-mono text-gray-400"
							title="sort experiments"
						>
							<option value="bpb">sort: bpb</option>
							<option value="name">sort: a-z</option>
							<option value="newest">sort: newest</option>
							<option value="oldest">sort: oldest</option>
							<option value="steps">sort: steps</option>
						</select>
					</div>
					{#if selectionMode}
						<div class="mb-2 flex items-center gap-2 text-[10px] font-mono text-gray-500">
							<span>{selectedBatchIds.length} selected</span>
							<button onclick={rerunSelectedBatch} disabled={running || importing || selectedBatchIds.length === 0} class="text-gray-400 hover:text-white disabled:opacity-40">
								rerun selected
							</button>
							<button onclick={deleteSelectedBatch} disabled={running || importing || selectedBatchIds.length === 0} class="text-red-400 hover:text-red-200 disabled:opacity-40">
								delete selected
							</button>
							<button onclick={clearBatchSelection} disabled={selectedBatchIds.length === 0} class="text-gray-500 hover:text-gray-300 disabled:opacity-40">
								clear selection
							</button>
						</div>
					{/if}
					<div class="flex-1 min-h-0 overflow-y-auto">
						<Leaderboard experiments={allExperiments} onSelect={selectExperiment}
							selected={selectedExp}
							sortMode={leaderboardSort}
							selectionEnabled={selectionMode}
							selectedIds={selectedBatchIds}
							onToggleBatchSelect={toggleBatchSelection} />
					</div>
					<div class="flex gap-3 mt-2 pt-2 border-t border-gray-800 shrink-0">
						<input bind:this={importInput} type="file" accept=".zip,application/zip" class="hidden" onchange={handleImportFile} />
						<button onclick={handleImportClick} disabled={running || importing} class="text-gray-500 hover:text-gray-300 disabled:opacity-40 text-xs font-mono">
							{importing ? 'importing...' : 'import'}
						</button>
						{#if experiments.length > 0}
							<button onclick={rerunAllExperiments} disabled={running || importing} class="text-gray-500 hover:text-gray-300 disabled:opacity-40 text-xs font-mono">rerun all</button>
							<button onclick={handleExport} class="text-gray-500 hover:text-gray-300 text-xs font-mono">export</button>
							<button onclick={handleClear} disabled={running} class="text-gray-500 hover:text-red-400 text-xs font-mono">clear</button>
						{/if}
					</div>
				</div>
			</div>
		</div>
		</div>
		{/if}
	{/if}

	{#if showClearModal}
		<div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
			<div class="bg-gray-900 border border-gray-700 rounded-lg p-6 max-w-sm space-y-4 font-mono">
				<h3 class="text-sm text-gray-200">clear all data?</h3>
				<p class="text-xs text-gray-400">this will delete all experiments, loss curves, inferences, and saved weights. this cannot be undone.</p>
				<div class="flex gap-2 justify-end">
					<button onclick={() => (showClearModal = false)} class="px-3 py-1.5 rounded bg-gray-800 text-gray-300 hover:bg-gray-700 text-sm">cancel</button>
					<button onclick={confirmClear} class="px-3 py-1.5 rounded bg-red-600 text-white hover:bg-red-500 text-sm">clear everything</button>
				</div>
			</div>
		</div>
	{/if}
</main>
