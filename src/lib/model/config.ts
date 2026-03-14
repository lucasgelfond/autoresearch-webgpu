import defaults from '$lib/defaults.json';

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

export type Range = { min?: number; max?: number };

export type ParamConstraints = {
	[K in keyof ExperimentConfig]?: Range;
} & {
	maxParams?: Range;
};

export const DEFAULT_CONSTRAINTS: ParamConstraints = {
	nLayer: { min: 1, max: 8 },
	nEmbd: { min: 32, max: 256 },
	nHead: { min: 1, max: 8 },
	mlpRatio: { min: 2, max: 6 },
	batchSize: { min: 4, max: 32 },
	seqLen: { min: 64, max: 256 },
	lr: { min: 0.00001, max: 0.01 },
	trainSeconds: { max: 30 },
	maxParams: { max: 400_000 },
};

/** Sensible defaults loaded from static/defaults.json */
export const DEFAULT_CONFIG: ExperimentConfig = defaults as ExperimentConfig;

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
