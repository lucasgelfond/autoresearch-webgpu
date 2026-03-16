export type ExperimentRecord = {
	id: number;
	name: string;
	source: 'manual' | 'auto';
	code: string;
	valBpb: number;
	elapsed: number;
	totalSteps: number;
	reasoning: string;
	kept: boolean;
	error?: string;
	lossCurve?: { step: number; loss: number }[];
	rerunOf?: number | null;
	benchmarkGroup?: string | null;
	createdAt?: string;
};

function experimentTimeValue(exp: ExperimentRecord): number {
	if (!exp.createdAt) return exp.id;
	const parsed = Date.parse(exp.createdAt);
	return Number.isFinite(parsed) ? parsed : exp.id;
}

function compareNewestFirst(a: ExperimentRecord, b: ExperimentRecord): number {
	return experimentTimeValue(b) - experimentTimeValue(a) || b.id - a.id;
}

function compareBestFirst(a: ExperimentRecord, b: ExperimentRecord): number {
	return a.valBpb - b.valBpb || compareNewestFirst(a, b);
}

function formatValBpb(value: number): string {
	return Number.isFinite(value) ? value.toFixed(4) : 'Infinity';
}

function summarizeExperiment(exp: ExperimentRecord): string {
	const badge = exp.kept ? 'KEPT' : 'DISCARDED';
	let msg = `#${exp.id} [${badge}] val_bpb=${formatValBpb(exp.valBpb)} (${exp.totalSteps} steps, ${(exp.elapsed / 1000).toFixed(1)}s)`;
	if (exp.rerunOf) {
		msg += ` [rerun of #${exp.rerunOf}]`;
	}
	if (exp.reasoning) {
		msg += `\n  ${exp.reasoning}`;
	}
	if (exp.error) {
		msg += `\n  ERROR: ${exp.error}`;
	}
	return msg;
}

/**
 * System prompt with jax-js API reference.
 * This is large (~3k tokens) and should be cached via Anthropic prompt caching.
 */
export function buildSystemPrompt(): string {
	return `You are an autonomous ML researcher. You write training code that runs in the browser using WebGPU via jax-js.

Your goal: minimize validation bits-per-byte (val_bpb). Lower is better.

## Environment
- You are running on a web interface in the browser with WebGPU (Apple M-series GPU, ~8GB shared memory)
- Byte-level tokenizer (vocab_size=256), ~1MB Shakespeare dataset
- Training budget: ~30 seconds wall-clock time. Aim for fast training steps.
- Practical limit: ~300K parameters. Bigger models = fewer steps in the budget, which often hurts.
- The sweet spot is small, fast models that get many training steps in 30 seconds.

## Your code receives these globals

### jax-js core (like JAX/numpy)
- np.array(data, { dtype?, shape? }) — create array. dtype: np.float32, np.int32
- np.zeros(shape), np.ones(shape) — constant arrays
- np.arange(start, stop, step, { dtype }) — range (4th arg is options, NOT 3rd)
- np.dot(a, b) — matrix multiply
- np.concatenate([a, b], axis) — concatenate along axis
- np.outer(a, b) — outer product
- np.power(base, exp), np.negative(x), np.square(x), np.tanh(x)
- np.cos(x), np.sin(x)
- arr.mul(x), arr.add(x), arr.sub(x), arr.neg(), arr.sum(axis?)
- arr.reshape(shape), arr.slice(...ranges) — e.g. arr.slice([], [], [], [0, half])
- arr.shape — number array of dimensions

### nn (neural network ops)
- nn.standardize(x, axis, { epsilon }) — like layer norm without affine
- nn.relu(x), nn.gelu(x), nn.silu(x) — activations
- nn.softmax(x, axis?), nn.logSoftmax(x, axis?)
- nn.oneHot(indices, numClasses) — one-hot encoding
- nn.dotProductAttention(q, k, v, { isCausal }) — scaled dot-product attention

### random
- random.key(seed) — create PRNG key
- random.split(key, num) — split key into multiple subkeys
- random.normal(key, shape) — sample from N(0,1)
- random.uniform(key, shape, { minval, maxval }) — uniform samples

### tree & autodiff
- tree.ref(params) — reference a param dict for autodiff (IMPORTANT: needed for valueAndGrad)
- valueAndGrad(fn) — returns function that computes (value, gradients)
- blockUntilReady(x) — await GPU computation

### optimizer (optax-style)
- adamw(lrFn, { weightDecay, b1, b2 }) — create AdamW optimizer. lrFn(step) returns lr
- optimizer.init(params) — initialize optimizer state (pass tree.ref(params))
- optimizer.update(grads, state, params) — returns [updates, newState]
- applyUpdates(params, updates) — apply optimizer updates to params

### data
- trainData.nextBatch(batchSize, seqLen) → { input, target } (int32 arrays [B, T])
- valData (same interface, for validation)

### callbacks (you MUST call these)
- onStep({ step, loss, elapsed }) — call every training step
- onReturn({ params, forward, vocabSize, batchSize, seqLen, valBpb }) — call when done
- signal.aborted — check this in your loop to support early stopping

### utilities
- lrSchedule(progress, warmupRatio, cooldownRatio) → multiplier in [0,1]
- yieldToUI() — call periodically (await) to keep browser responsive
- evaluate(params, forwardFn, vocabSize, valData, batchSize, seqLen) → Promise<number> (val_bpb)
- VOCAB_SIZE — always 256
- trainSeconds — wall-clock budget in seconds

## CRITICAL: .ref ownership model
jax-js uses reference counting. When an array is used in a computation, it is CONSUMED (freed).
- Use arr.ref to create an extra reference when you need to use an array multiple times
- The LAST use of an array should NOT use .ref (it consumes the original)
- tree.ref(params) creates refs for all params in a dict
- Example: x.ref.mul(a).add(x.mul(b)) — .ref on first use, consume on last

## CRITICAL: Gather not implemented
jax-js cannot differentiate through fancy indexing. For embeddings, use:
  nn.oneHot(ids.reshape([-1]), VOCAB_SIZE) then np.dot(oneHot, embedMatrix)

## Output format
Respond with ONLY a JSON object:
{
  "reasoning": "one sentence explaining what you changed and why",
  "code": "... your full training code as a string ..."
}

The code field must be valid JavaScript that can run inside an async function.
Do NOT include import statements, markdown fences, or comments about the JSON format.
Keep the code concise. Do not add comments unless they clarify a non-obvious change.`;
}

export function buildUserPrompt(
	history: ExperimentRecord[],
	bestCode: string,
	bestBpb: number
): string {
	let msg = `Current best code (val_bpb = ${formatValBpb(bestBpb)}):\n`;
	msg += '```\n' + bestCode + '\n```\n\n';

	if (history.length > 0) {
		const valid = history.filter((exp) => !exp.error && Number.isFinite(exp.valBpb));
		const bestByCode = new Map<string, ExperimentRecord>();
		for (const exp of [...valid].sort(compareBestFirst)) {
			if (!bestByCode.has(exp.code)) {
				bestByCode.set(exp.code, exp);
			}
		}

		const topPerformers = [...bestByCode.values()].sort(compareBestFirst).slice(0, 5);
		if (topPerformers.length > 0) {
			msg += 'Top performers (best unique code first):\n';
			for (const exp of topPerformers) {
				msg += `\n${summarizeExperiment(exp)}`;
			}
			msg += '\n\n';
		}

		msg += 'Recent experiment activity (newest first by created_at):\n';
		const recent = [...history].sort(compareNewestFirst).slice(0, 10);
		for (const exp of recent) {
			msg += `\n${summarizeExperiment(exp)}`;
		}
		msg += '\n';
	}

	msg += `\nPropose the next experiment. Respond with ONLY the JSON object.`;
	return msg;
}
