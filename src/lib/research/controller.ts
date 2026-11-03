import { DataLoader } from '../data/loader';
import { executeTrainCode, type RunResult } from './sandbox';
import type { StepMetrics } from '../prepare';
import { petname } from '../petname';
import { insertExperiment, insertLossCurve, updateWeightsPath } from '../db';
import { saveWeights } from '../weights';
import { buildSystemPrompt, buildUserPrompt, type ExperimentRecord } from './prompt';
import { BASELINE_CODE } from './baseline';
import { parseResponse, extractPartial } from './parse';
import type { ResearchProvider } from './providers/types';

export type ResearchCallbacks = {
	onExperimentStart?: (code: string, reasoning: string) => void;
	onStep?: (metrics: StepMetrics) => void;
	onExperimentDone?: (record: ExperimentRecord) => void;
	onError?: (error: string) => void;
	onCodeStream?: (text: string) => void;
	onReasoningStream?: (text: string) => void;
};

export class ResearchController {
	history: ExperimentRecord[] = [];
	bestCode: string = BASELINE_CODE;
	bestBpb: number = Infinity;
	running: boolean = false;
	lastError = '';
	trainSeconds = 30;
	private stopRequested = false;
	private runAbort: AbortController | null = null;
	private generateAbort: AbortController | null = null;

	constructor(private provider: ResearchProvider) {}

	setProvider(provider: ResearchProvider) {
		this.provider = provider;
	}

	stop() {
		this.stopRequested = true;
		this.generateAbort?.abort();
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
				this.bestCode,
				'Baseline run with default architecture.',
				trainData, valData, callbacks
			);
		}

		while (!this.stopRequested) {
			const proposal = await this.getNextCode(callbacks);
			if (!proposal) {
				callbacks.onError?.(this.lastError || 'Failed to get next code from the model.');
				break;
			}
			await this.runExperiment(
				proposal.code,
				proposal.reasoning,
				trainData, valData, callbacks
			);
		}

		this.running = false;
	}

	private async runExperiment(
		code: string,
		reasoning: string,
		trainData: DataLoader,
		valData: DataLoader,
		callbacks: ResearchCallbacks
	) {
		callbacks.onExperimentStart?.(code, reasoning);

		this.runAbort = new AbortController();
		const lossCurve: { step: number; loss: number }[] = [];

		const result = await executeTrainCode(code, trainData, valData, this.trainSeconds, {
			signal: this.runAbort.signal,
			onStep(m) {
				lossCurve.push({ step: m.step, loss: m.loss });
				callbacks.onStep?.(m);
			}
		});
		this.runAbort = null;

		const kept = result.valBpb < this.bestBpb && !result.error;
		if (kept) {
			this.bestBpb = result.valBpb;
			this.bestCode = code;
		}

		const expName = petname();
		const dbId = await insertExperiment({
			name: expName,
			source: 'auto',
			code,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning,
			kept,
			lossCurve,
			error: result.error,
		});

		await insertLossCurve(dbId, lossCurve);

		if (result.params && Object.keys(result.params).length > 0) {
			(async () => {
				try {
					const weightsPath = await saveWeights(dbId, result.params);
					await updateWeightsPath(dbId, weightsPath);
				} catch (e) { console.error('Failed to save weights:', e); }
			})();
		}

		const record: ExperimentRecord = {
			id: dbId,
			name: expName,
			source: 'auto',
			code,
			valBpb: result.valBpb,
			elapsed: result.elapsed,
			totalSteps: result.totalSteps,
			reasoning,
			kept,
			error: result.error,
			lossCurve
		};

		this.history.push(record);
		callbacks.onExperimentDone?.(record);
	}

	private async getNextCode(callbacks: ResearchCallbacks): Promise<{ code: string; reasoning: string } | null> {
		// Greedy first; small models sometimes repeat a past experiment or emit unparseable
		// output, so retry with sampling before giving up.
		for (const temperature of [0, 0.7, 0.7]) {
			if (this.stopRequested) return null;
			const proposal = await this.propose(callbacks, temperature);
			if (proposal && !this.history.some((e) => e.code === proposal.code)) return proposal;
			if (proposal) this.lastError = 'Model repeated a previous experiment';
		}
		return null;
	}

	private async propose(callbacks: ResearchCallbacks, temperature: number): Promise<{ code: string; reasoning: string } | null> {
		const systemPrompt = buildSystemPrompt();
		const userPrompt = buildUserPrompt(this.history, this.bestCode, this.bestBpb);

		this.generateAbort = new AbortController();
		let fullText = '';
		try {
			await this.provider.generate({
				systemPrompt,
				userPrompt,
				temperature,
				signal: this.generateAbort.signal,
				onToken: (chunk) => {
					fullText += chunk;
					const extracted = extractPartial(fullText);
					if (extracted.code) callbacks.onCodeStream?.(extracted.code);
					if (extracted.reasoning) callbacks.onReasoningStream?.(extracted.reasoning);
				}
			});
		} catch (e) {
			if (!this.stopRequested) this.lastError = `Generation failed: ${e}`;
			return null;
		} finally {
			this.generateAbort = null;
		}

		if (!fullText) {
			this.lastError = 'Empty response from model';
			return null;
		}
		const result = parseResponse(fullText);
		if (!result) {
			this.lastError = 'Could not parse model response';
			console.warn('Unparseable model response:', fullText);
		}
		return result;
	}
}
