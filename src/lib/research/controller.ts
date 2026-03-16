import { DataLoader } from '../data/loader';
import { executeTrainCode, type RunResult } from './sandbox';
import type { StepMetrics } from '../prepare';
import { petname } from '../petname';
import { insertExperiment, insertLossCurve, updateWeightsPath } from '../db';
import { saveWeights } from '../weights';
import { buildSystemPrompt, buildUserPrompt, type ExperimentRecord } from './prompt';
import { BASELINE_CODE } from './baseline';
import { parseClaudeResponse } from './parse';
import type { ResearchEndpointProfile } from './providers';

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
	profile: ResearchEndpointProfile | null = null;
	private stopRequested = false;
	private runAbort: AbortController | null = null;
	private fetchAbort: AbortController | null = null;

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
				this.bestCode,
				'Baseline run with default architecture.',
				trainData, valData, callbacks
			);
		}

		while (!this.stopRequested) {
			const proposal = await this.getNextCode(callbacks);
			if (!proposal) {
				callbacks.onError?.(this.lastError || 'Failed to get next code from Claude.');
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
		const systemPrompt = buildSystemPrompt();
		const userPrompt = buildUserPrompt(this.history, this.bestCode, this.bestBpb);

		this.fetchAbort = new AbortController();
		try {
			const response = await fetch('/api/research', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					systemPrompt,
					userPrompt,
					stream: true,
					profile: this.profile,
				}),
				signal: this.fetchAbort.signal
			});

			if (!response.ok) {
				const err = await response.text();
				this.lastError = `API ${response.status}: ${err.slice(0, 200)}`;
				return null;
			}

			// Parse SSE stream from Anthropic
			const fullText = await this.consumeStream(response, callbacks);
			if (!fullText) {
				this.lastError = 'Empty response from Claude';
				return null;
			}

			return this.parseResponse(fullText);
		} catch (e) {
			if (this.stopRequested) return null;
			this.lastError = `Fetch failed: ${e}`;
			return null;
		} finally {
			this.fetchAbort = null;
		}
	}

	private async consumeStream(response: Response, callbacks: ResearchCallbacks): Promise<string> {
		const reader = response.body!.getReader();
		const decoder = new TextDecoder();
		let fullText = '';
		let buffer = '';

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() || '';

			for (const line of lines) {
				if (!line.startsWith('data: ')) continue;
				const data = line.slice(6);
				if (data === '[DONE]') continue;

				try {
					const event = JSON.parse(data);
					if (event.type === 'text_delta' && event.text) {
						const chunk = event.text;
						fullText += chunk;

						// Try to extract streaming code from the partial JSON
						const extracted = this.extractStreamingCode(fullText);
						if (extracted.code) {
							callbacks.onCodeStream?.(extracted.code);
						}
						if (extracted.reasoning) {
							callbacks.onReasoningStream?.(extracted.reasoning);
						}
					}
				} catch {}
			}
		}

		return fullText;
	}

	/** Try to extract code and reasoning from partial JSON as it streams in. */
	private extractStreamingCode(partial: string): { code?: string; reasoning?: string } {
		const result: { code?: string; reasoning?: string } = {};

		// Try to find reasoning field
		const reasoningMatch = partial.match(/"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"/);
		if (reasoningMatch) {
			try { result.reasoning = JSON.parse('"' + reasoningMatch[1] + '"'); } catch {}
		}

		// Try to find code field — it may be incomplete
		const codeStart = partial.indexOf('"code"');
		if (codeStart === -1) return result;

		// Find the opening quote of the code value
		const colonAfterCode = partial.indexOf(':', codeStart + 6);
		if (colonAfterCode === -1) return result;

		const quoteStart = partial.indexOf('"', colonAfterCode);
		if (quoteStart === -1) return result;

		// Extract the code value, handling escaped characters
		// Walk through the string handling escapes
		let code = '';
		let i = quoteStart + 1;
		while (i < partial.length) {
			if (partial[i] === '\\' && i + 1 < partial.length) {
				const next = partial[i + 1];
				if (next === '"') { code += '"'; i += 2; }
				else if (next === '\\') { code += '\\'; i += 2; }
				else if (next === 'n') { code += '\n'; i += 2; }
				else if (next === 't') { code += '\t'; i += 2; }
				else if (next === 'r') { code += '\r'; i += 2; }
				else { code += partial[i]; i++; }
			} else if (partial[i] === '"') {
				// End of string
				break;
			} else {
				code += partial[i];
				i++;
			}
		}

		if (code.length > 0) {
			result.code = code;
		}

		return result;
	}

	private parseResponse(text: string): { code: string; reasoning: string } | null {
		const result = parseClaudeResponse(text);
		if (!result) this.lastError = 'Could not parse Claude response';
		return result;
	}
}
