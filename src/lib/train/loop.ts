import { blockUntilReady, valueAndGrad, numpy as np, random, tree, nn } from '@jax-js/jax';
import { adamw, applyUpdates, type OptState } from '@jax-js/optax';
import type { ExperimentConfig } from '../model/config';
import { initParams, lossFn, forward, type Params } from '../model/gpt';
import { DataLoader } from '../data/loader';
import { lrMultiplier } from './schedule';

export type StepMetrics = {
	step: number;
	loss: number;
	elapsed: number;
	lr: number;
};

export type RunResult = {
	valBpb: number;
	finalLoss: number;
	totalSteps: number;
	elapsed: number;
	params: Params;
};

export type TrainCallbacks = {
	onStep?: (metrics: StepMetrics) => void;
	onDone?: (result: RunResult) => void;
};

function yieldToUI(): Promise<void> {
	return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

export async function trainRun(
	config: ExperimentConfig,
	trainData: DataLoader,
	valData: DataLoader,
	callbacks: TrainCallbacks = {}
): Promise<RunResult> {
	const key = random.key(42);
	let params = initParams(config, key);
	await blockUntilReady(params);

	const baseLr = config.lr;
	const optimizer = adamw((step: number) => {
		const progress = Math.min(elapsed / (config.trainSeconds * 1000), 1);
		return -baseLr * lrMultiplier(progress, config.warmupRatio, config.cooldownRatio);
	}, { weightDecay: config.weightDecay, b1: 0.9, b2: 0.95 });

	let optState = optimizer.init(tree.ref(params));
	let updates: Params;

	const lossGrad = valueAndGrad((p: Params, input: np.Array, target: np.Array) => {
		return lossFn(p, config, input, target);
	});

	let step = 0;
	let elapsed = 0;
	let lastLoss = 0;
	const t0 = performance.now();

	while (elapsed < config.trainSeconds * 1000) {
		const stepStart = performance.now();

		const batch = trainData.nextBatch(config.batchSize, config.seqLen);
		const [lossVal, grads] = lossGrad(tree.ref(params), batch.input, batch.target);

		[updates, optState] = optimizer.update(grads, optState, tree.ref(params));
		params = applyUpdates(params, updates) as Params;

		await blockUntilReady(params);
		const lossNumber = (await lossVal.jsAsync()) as number;
		lastLoss = lossNumber;

		elapsed = performance.now() - t0;
		const progress = Math.min(elapsed / (config.trainSeconds * 1000), 1);
		const lr = baseLr * lrMultiplier(progress, config.warmupRatio, config.cooldownRatio);

		step++;

		callbacks.onStep?.({
			step,
			loss: lossNumber,
			elapsed,
			lr
		});

		// Yield to UI every step so the browser stays responsive
		if (step % 1 === 0) {
			await yieldToUI();
		}

		// Bail on NaN
		if (isNaN(lossNumber)) {
			break;
		}
	}

	// Evaluate val BPB
	const valBpb = await evaluateBpb(params, config, valData);

	const result: RunResult = {
		valBpb,
		finalLoss: lastLoss,
		totalSteps: step,
		elapsed,
		params
	};

	callbacks.onDone?.(result);
	return result;
}

async function evaluateBpb(
	params: Params,
	config: ExperimentConfig,
	valData: DataLoader
): Promise<number> {
	valData.reset();

	const evalTokens = Math.min(valData.length, 100_000);
	const tokensPerBatch = config.batchSize * config.seqLen;
	const numSteps = Math.max(1, Math.floor(evalTokens / tokensPerBatch));

	let totalNats = 0;
	let totalTokens = 0;

	for (let i = 0; i < numSteps; i++) {
		const batch = valData.nextBatch(config.batchSize, config.seqLen);
		const logits = forward(params, config, batch.input);
		const logProbs = nn.logSoftmax(logits, -1);
		const targets = nn.oneHot(batch.target.reshape([-1]), config.vocabSize);
		const flatLogProbs = logProbs.reshape([-1, config.vocabSize]);

		const nll = flatLogProbs.mul(targets).sum().neg();
		await blockUntilReady(nll);
		totalNats += (await nll.jsAsync()) as number;
		totalTokens += config.batchSize * config.seqLen;
	}

	// For byte-level tokenizer, 1 token = 1 byte, so BPB = nats / (ln2 * bytes)
	return totalNats / (Math.LN2 * totalTokens);
}
