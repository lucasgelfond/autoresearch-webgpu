import { numpy as np, nn, blockUntilReady } from '@jax-js/jax';
import type { ExperimentConfig } from './model/config';
import { forward, type Params } from './model/gpt';
import { decode } from './data/tokenizer';

export async function sampleText(
	params: Params,
	config: ExperimentConfig,
	prompt: string = '',
	maxTokens: number = 200,
	temperature: number = 0.8
): Promise<string> {
	const encoder = new TextEncoder();
	const promptBytes = prompt ? Array.from(encoder.encode(prompt)) : [0];

	const tokens: number[] = [...promptBytes];

	for (let i = 0; i < maxTokens; i++) {
		// Take the last seqLen tokens as context
		const contextStart = Math.max(0, tokens.length - config.seqLen);
		const context = tokens.slice(contextStart);

		const inputIds = np.array(context, { dtype: np.int32 }).reshape([1, context.length]);
		const logits = forward(params, config, inputIds);

		// Get logits for the last position
		const lastLogits = logits.slice(0, [-1]).reshape([config.vocabSize]);

		// Temperature scaling + softmax sampling
		const scaled = lastLogits.mul(1 / temperature);
		const probs = nn.softmax(scaled);
		await blockUntilReady(probs);

		const probsArr = (await probs.jsAsync()) as number[];
		const nextToken = sampleFromProbs(probsArr);
		tokens.push(nextToken);
	}

	return decode(new Uint8Array(tokens));
}

function sampleFromProbs(probs: number[]): number {
	const r = Math.random();
	let cumulative = 0;
	for (let i = 0; i < probs.length; i++) {
		cumulative += probs[i];
		if (r < cumulative) return i;
	}
	return probs.length - 1;
}
