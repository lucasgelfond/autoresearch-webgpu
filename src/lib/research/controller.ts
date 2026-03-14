import type { ExperimentConfig, ParamConstraints } from '../model/config';
import { DEFAULT_CONFIG } from '../model/config';
import { DataLoader } from '../data/loader';
import { trainRun, type StepMetrics, type RunResult } from '../train/loop';
import { sampleText } from '../sample';
import { petname } from '../petname';
import { insertExperiment, insertInference, insertLossCurve, updateWeightsPath } from '../db';
import { saveWeights } from '../weights';
import { buildSystemPrompt, buildUserPrompt, type ExperimentRecord } from './prompt';

export type ResearchCallbacks = {
	onExperimentStart?: (config: ExperimentConfig, reasoning: string) => void;
	onStep?: (metrics: StepMetrics) => void;
	onExperimentDone?: (record: ExperimentRecord) => void;
	onError?: (error: string) => void;
};

export class ResearchController {
	history: ExperimentRecord[] = [];
	bestConfig: ExperimentConfig;
	bestBpb: number = Infinity;
	running: boolean = false;
	lastError = '';
	constraints?: ParamConstraints;
	private stopRequested = false;
	private runAbort: AbortController | null = null;
	private fetchAbort: AbortController | null = null;

	constructor() {
		this.bestConfig = { ...DEFAULT_CONFIG };
	}

	stop() {
		this.stopRequested = true;
		this.fetchAbort?.abort();
		this.runAbort?.abort();
	}

	stopCurrentRun() {
		this.runAbort?.abort();
	}

	async run(
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks = {}
	) {
		this.running = true;
		this.stopRequested = false;

		if (this.history.length === 0) {
			await this.runExperiment(
				this.bestConfig,
				'Baseline run with default config.',
				trainData,
				valData,
				callbacks
			);
		}

		while (!this.stopRequested) {
			const proposal = await this.getNextConfig();
			if (!proposal) {
				callbacks.onError?.(this.lastError || 'Failed to get next config from Claude.');
				break;
			}

			await this.runExperiment(
				proposal.config,
				proposal.reasoning,
				trainData,
				valData,
				callbacks
			);
		}

		this.running = false;
	}

	private async runExperiment(
		config: ExperimentConfig,
		reasoning: string,
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks
	) {
		callbacks.onExperimentStart?.(config, reasoning);

		trainData.reset();
		this.runAbort = new AbortController();
		const lossCurve: { step: number; loss: number }[] = [];
		const result = await trainRun(config, trainData, valData, {
			signal: this.runAbort.signal,
			onStep(m) {
				lossCurve.push({ step: m.step, loss: m.loss });
				callbacks.onStep?.(m);
			}
		});
		this.runAbort = null;

		const kept = result.valBpb < this.bestBpb;
		if (kept) {
			this.bestBpb = result.valBpb;
			this.bestConfig = { ...config };
		}

		const expName = petname();
		const dbId = await insertExperiment({
			name: expName,
			source: 'auto',
			config,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning,
			kept,
			lossCurve
		});

		// Save loss curve to normalized table
		await insertLossCurve(dbId, lossCurve);

		// Save weights to OPFS
		try {
			const weightsPath = await saveWeights(dbId, result.params);
			await updateWeightsPath(dbId, weightsPath);
		} catch (_) {}

		// Generate sample in background — don't block next experiment
		const bgParams = result.params;
		const bgConfig = config;
		(async () => {
			try {
				const output = await sampleText(bgParams, bgConfig, '', 200, 0.8);
				await insertInference({ experimentId: dbId, prompt: '', output, temperature: 0.8 });
			} catch (_) {}
		})();

		const record: ExperimentRecord = {
			id: dbId,
			name: expName,
			source: 'auto',
			config,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning,
			kept,
			sampleText: '',
			lossCurve
		};

		this.history.push(record);
		callbacks.onExperimentDone?.(record);
	}

	private async getNextConfig(): Promise<{ config: ExperimentConfig; reasoning: string } | null> {
		const systemPrompt = buildSystemPrompt();
		const userPrompt = buildUserPrompt(this.history, this.bestConfig, this.bestBpb, this.constraints);

		this.fetchAbort = new AbortController();
		try {
			const response = await fetch('/api/research', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ systemPrompt, userPrompt }),
				signal: this.fetchAbort.signal
			});

			if (!response.ok) {
				const err = await response.text();
				console.error('Research API error:', response.status, err);
				this.lastError = `API ${response.status}: ${err.slice(0, 200)}`;
				return null;
			}

			const data = await response.json();
			if (data.error) {
				console.error('Research API error:', data.error);
				this.lastError = `API error: ${data.error}`;
				return null;
			}

			return {
				config: { ...this.bestConfig, ...data.config, vocabSize: 256 },
				reasoning: data.reasoning || 'No reasoning provided.'
			};
		} catch (e) {
			if (this.stopRequested) return null;
			console.error('Research API fetch failed:', e);
			this.lastError = `Fetch failed: ${e}`;
			return null;
		} finally {
			this.fetchAbort = null;
		}
	}
}
