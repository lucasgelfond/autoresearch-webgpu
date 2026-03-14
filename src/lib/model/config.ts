export type Activation = 'relu_sq' | 'gelu' | 'silu';

export type ExperimentConfig = {
	// architecture
	nLayer: number;
	nEmbd: number;
	nHead: number;
	mlpRatio: number;
	activation: Activation;
	useRoPE: boolean;
	softcapValue: number; // 0 = disabled

	// optimization
	lr: number;
	weightDecay: number;
	warmupRatio: number;
	cooldownRatio: number;
	batchSize: number;
	seqLen: number;

	// training
	trainSeconds: number;

	// data
	vocabSize: number;
};

/** Sensible defaults for M4 WebGPU (~1.5M params). */
export const DEFAULT_CONFIG: ExperimentConfig = {
	nLayer: 4,
	nEmbd: 128,
	nHead: 4,
	mlpRatio: 4,
	activation: 'relu_sq',
	useRoPE: true,
	softcapValue: 15,

	lr: 3e-4,
	weightDecay: 0.1,
	warmupRatio: 0.1,
	cooldownRatio: 0.3,
	batchSize: 16,
	seqLen: 256,

	trainSeconds: 60,

	vocabSize: 2048
};

/** Estimate total parameter count for a config. */
export function estimateParams(c: ExperimentConfig): number {
	const { nLayer, nEmbd, nHead, mlpRatio, vocabSize } = c;
	const headDim = nEmbd / nHead;

	// embedding + unembedding
	const embed = vocabSize * nEmbd;
	const unembed = nEmbd * vocabSize;

	// per layer: attn (q,k,v,proj) + mlp (up, down) + 2 norm weights
	const attnQKV = 3 * nEmbd * (nHead * headDim);
	const attnProj = nEmbd * nEmbd;
	const mlpUp = nEmbd * (nEmbd * mlpRatio);
	const mlpDown = (nEmbd * mlpRatio) * nEmbd;
	const norms = 2 * nEmbd;
	const perLayer = attnQKV + attnProj + mlpUp + mlpDown + norms;

	// final norm
	const finalNorm = nEmbd;

	return embed + unembed + nLayer * perLayer + finalNorm;
}
