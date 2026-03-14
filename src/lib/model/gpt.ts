import { numpy as np, nn, random } from '@jax-js/jax';
import type { ExperimentConfig, Activation } from './config';

export type Params = { [key: string]: np.Array };

export function initParams(config: ExperimentConfig, key: np.Array): Params {
	const { nLayer, nEmbd, nHead, mlpRatio, vocabSize } = config;
	const headDim = nEmbd / nHead;
	const mlpHidden = nEmbd * mlpRatio;
	const params: Params = {};

	const numKeys = 3 + nLayer * 8;
	const keys = random.split(key, numKeys);
	let ki = 0;
	let k: np.Array;

	k = keys.slice(ki++);
	params['embed'] = random.normal(k, [vocabSize, nEmbd]).mul(1.0);

	for (let i = 0; i < nLayer; i++) {
		const s = Math.sqrt(3) * Math.pow(nEmbd, -0.5);
		const prefix = `layer${i}`;

		k = keys.slice(ki++);
		params[`${prefix}.attn.wq`] = random.uniform(k, [nEmbd, nHead * headDim], {
			minval: -s,
			maxval: s
		});

		k = keys.slice(ki++);
		params[`${prefix}.attn.wk`] = random.uniform(k, [nEmbd, nHead * headDim], {
			minval: -s,
			maxval: s
		});

		k = keys.slice(ki++);
		params[`${prefix}.attn.wv`] = random.uniform(k, [nEmbd, nHead * headDim], {
			minval: -s,
			maxval: s
		});

		k = keys.slice(ki++);
		params[`${prefix}.attn.wout`] = np.zeros([nEmbd, nEmbd]);

		k = keys.slice(ki++);
		params[`${prefix}.norm1`] = np.ones([nEmbd]);

		k = keys.slice(ki++);
		params[`${prefix}.norm2`] = np.ones([nEmbd]);

		k = keys.slice(ki++);
		params[`${prefix}.mlp.up`] = random.uniform(k, [nEmbd, mlpHidden], {
			minval: -s,
			maxval: s
		});

		k = keys.slice(ki++);
		params[`${prefix}.mlp.down`] = np.zeros([mlpHidden, nEmbd]);
	}

	params['final_norm'] = np.ones([nEmbd]);

	k = keys.slice(ki++);
	params['unembed'] = random.normal(k, [nEmbd, vocabSize]).mul(0.001);

	return params;
}

function rmsNorm(x: np.Array, weight: np.Array): np.Array {
	return nn.standardize(x, -1, { epsilon: 1e-6 }).mul(weight);
}

function ropeFreqs(seqLen: number, headDim: number): [np.Array, np.Array] {
	const halfDim = headDim / 2;
	const freqExponents = np.arange(0, halfDim).mul(2 / headDim);
	const invFreq = np.power(10000, np.negative(freqExponents));
	const positions = np.arange(seqLen).astype(np.float32);
	const angles = np.outer(positions, invFreq);
	return [np.cos(angles), np.sin(angles)];
}

function applyRoPE(x: np.Array, cos: np.Array, sin: np.Array): np.Array {
	const half = x.shape[3] / 2;
	const x1 = x.slice([], [], [], [0, half]);
	const x2 = x.slice([], [], [], [half]);
	const c = cos.reshape([1, -1, 1, half]);
	const s = sin.reshape([1, -1, 1, half]);
	return np.concatenate([x1.mul(c).sub(x2.mul(s)), x1.mul(s).add(x2.mul(c))], -1);
}

function activate(x: np.Array, activation: Activation): np.Array {
	switch (activation) {
		case 'relu_sq':
			return np.square(nn.relu(x));
		case 'gelu':
			return nn.gelu(x);
		case 'silu':
			return nn.silu(x);
	}
}

export function forward(
	params: Params,
	config: ExperimentConfig,
	inputIds: np.Array
): np.Array {
	const { nLayer, nHead, nEmbd, mlpRatio, activation, useRoPE, softcapValue } = config;
	const headDim = nEmbd / nHead;
	const [_batch, seqLen] = inputIds.shape;

	let x = params['embed'].slice(inputIds);

	let ropeCos: np.Array | null = null;
	let ropeSin: np.Array | null = null;
	if (useRoPE) {
		[ropeCos, ropeSin] = ropeFreqs(seqLen, headDim);
	}

	for (let i = 0; i < nLayer; i++) {
		const prefix = `layer${i}`;

		const normed = rmsNorm(x, params[`${prefix}.norm1`]);

		let q = np.dot(normed, params[`${prefix}.attn.wq`]).reshape([-1, seqLen, nHead, headDim]);
		let k = np.dot(normed, params[`${prefix}.attn.wk`]).reshape([-1, seqLen, nHead, headDim]);
		const v = np.dot(normed, params[`${prefix}.attn.wv`]).reshape([-1, seqLen, nHead, headDim]);

		if (useRoPE && ropeCos && ropeSin) {
			q = applyRoPE(q, ropeCos, ropeSin);
			k = applyRoPE(k, ropeCos, ropeSin);
		}

		const attnOut = nn.dotProductAttention(q, k, v, { isCausal: true });
		const projected = np.dot(attnOut.reshape([-1, seqLen, nEmbd]), params[`${prefix}.attn.wout`]);
		x = x.add(projected);

		const normed2 = rmsNorm(x, params[`${prefix}.norm2`]);
		let h = np.dot(normed2, params[`${prefix}.mlp.up`]);
		h = activate(h, activation);
		h = np.dot(h, params[`${prefix}.mlp.down`]);
		x = x.add(h);
	}

	x = rmsNorm(x, params['final_norm']);
	let logits = np.dot(x, params['unembed']);

	if (softcapValue > 0) {
		logits = np.tanh(logits.mul(1 / softcapValue)).mul(softcapValue);
	}

	return logits;
}

export function lossFn(
	params: Params,
	config: ExperimentConfig,
	inputIds: np.Array,
	targetIds: np.Array
): np.Array {
	const logits = forward(params, config, inputIds);
	const batchSize = targetIds.shape[0];
	const seqLen = targetIds.shape[1];

	const logProbs = nn.logSoftmax(logits, -1);
	const targets = nn.oneHot(targetIds.reshape([-1]), config.vocabSize);
	const flatLogProbs = logProbs.reshape([-1, config.vocabSize]);

	return flatLogProbs.mul(targets).sum().mul(-1 / (batchSize * seqLen));
}
