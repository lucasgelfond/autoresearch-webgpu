import {
	pipeline,
	TextStreamer,
	InterruptableStoppingCriteria,
	type TextGenerationPipeline,
	type ProgressInfo
} from '@huggingface/transformers';
import type { GenerateOptions, ResearchProvider } from './types';

export async function isWebGpuAvailable(): Promise<boolean> {
	if (typeof navigator === 'undefined' || !navigator.gpu) return false;
	try {
		return (await navigator.gpu.requestAdapter()) !== null;
	} catch {
		return false;
	}
}

export class WebGpuProvider implements ResearchProvider {
	private generator: TextGenerationPipeline | null = null;

	constructor(readonly modelId: string) {}

	async load(onProgress?: (info: ProgressInfo) => void) {
		const load = (dtype: 'q4f16' | 'q4') =>
			pipeline('text-generation', this.modelId, {
				device: 'webgpu',
				dtype,
				progress_callback: onProgress
			});
		try {
			this.generator = await load('q4f16');
		} catch (e) {
			console.warn('q4f16 load failed, retrying with q4:', e);
			this.generator = await load('q4');
		}
	}

	async dispose() {
		await this.generator?.dispose();
		this.generator = null;
	}

	async generate({ systemPrompt, userPrompt, onToken, signal, temperature = 0 }: GenerateOptions): Promise<string> {
		if (!this.generator) throw new Error('Model not loaded');
		const { tokenizer } = this.generator;

		const prompt = tokenizer.apply_chat_template(
			[
				{ role: 'system', content: systemPrompt },
				{ role: 'user', content: userPrompt }
			],
			{ tokenize: false, add_generation_prompt: true }
		);

		const stopping = new InterruptableStoppingCriteria();
		const onAbort = () => stopping.interrupt();
		signal?.addEventListener('abort', onAbort);

		let text = '';
		const streamer = new TextStreamer(tokenizer, {
			skip_prompt: true,
			skip_special_tokens: true,
			callback_function: (chunk: string) => {
				text += chunk;
				onToken?.(chunk);
			}
		});

		try {
			await this.generator(prompt, {
				max_new_tokens: 4096,
				do_sample: temperature > 0,
				temperature,
				streamer,
				stopping_criteria: stopping
			});
		} finally {
			signal?.removeEventListener('abort', onAbort);
		}

		if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
		return text;
	}
}
