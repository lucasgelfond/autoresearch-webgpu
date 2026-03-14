import type { ExperimentConfig } from '../model/config';
import { DEFAULT_CONFIG } from '../model/config';
import { DataLoader } from '../data/loader';
import { trainRun, type StepMetrics, type RunResult } from '../train/loop';
import { sampleText } from '../sample';
import { buildSystemPrompt, buildUserPrompt, type ExperimentRecord } from './prompt';

export type ResearchCallbacks = {
	onExperimentStart?: (id: number, config: ExperimentConfig, reasoning: string) => void;
	onStep?: (metrics: StepMetrics) => void;
	onExperimentDone?: (record: ExperimentRecord) => void;
	onError?: (error: string) => void;
};

export class ResearchController {
	history: ExperimentRecord[] = [];
	bestConfig: ExperimentConfig;
	bestBpb: number = Infinity;
	running: boolean = false;

	nextId = 1;
	lastError = '';
	private stopRequested = false;

	constructor() {
		this.bestConfig = { ...DEFAULT_CONFIG };
	}

	stop() {
		this.stopRequested = true;
	}

	async run(
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks = {}
	) {
		this.running = true;
		this.stopRequested = false;

		// Run first experiment with defaults if no history
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
		const id = this.nextId++;
		callbacks.onExperimentStart?.(id, config, reasoning);

		trainData.reset();
		const lossCurve: { step: number; loss: number }[] = [];
		const result = await trainRun(config, trainData, valData, {
			onStep(m) {
				lossCurve.push({ step: m.step, loss: m.loss });
				callbacks.onStep?.(m);
			}
		});

		const kept = result.valBpb < this.bestBpb;
		if (kept) {
			this.bestBpb = result.valBpb;
			this.bestConfig = { ...config };
		}

		let generatedSample = '';
		try {
			generatedSample = await sampleText(result.params, config, '', 200, 0.8);
		} catch (_) {}

		const record: ExperimentRecord = {
			id,
			config,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning,
			kept,
			sampleText: generatedSample,
			lossCurve
		};

		this.history.push(record);
		callbacks.onExperimentDone?.(record);
	}

	private async getNextConfig(): Promise<{ config: ExperimentConfig; reasoning: string } | null> {
		const systemPrompt = buildSystemPrompt();
		const userPrompt = buildUserPrompt(this.history, this.bestConfig, this.bestBpb);

		try {
			const response = await fetch('/api/research', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ systemPrompt, userPrompt })
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
			console.error('Research API fetch failed:', e);
			this.lastError = `Fetch failed: ${e}`;
			return null;
		}
	}
}
