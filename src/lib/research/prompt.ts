import type { ExperimentConfig } from '../model/config';

export type ExperimentRecord = {
	id: number;
	config: ExperimentConfig;
	valBpb: number;
	elapsed: number;
	totalSteps: number;
	reasoning: string;
	kept: boolean;
};

export function buildSystemPrompt(): string {
	return `You are an autonomous ML researcher. You are training small GPT language models
in the browser using WebGPU on an Apple M4. Your goal is to minimize validation
bits-per-byte (val_bpb) — lower is better.

The model is a transformer with:
- Byte-level tokenizer (vocab_size=256)
- RoPE positional embeddings
- RMSNorm
- Causal self-attention (standard, not flash)
- Configurable MLP activation (relu², gelu, silu)
- Optional softcap on logits

You control these parameters via a JSON config:
- nLayer (2-8): number of transformer layers
- nEmbd (64-256): embedding dimension (must be divisible by nHead)
- nHead (2-8): number of attention heads
- mlpRatio (2-6): MLP hidden dim = nEmbd * mlpRatio
- activation: "relu_sq" | "gelu" | "silu"
- useRoPE: boolean
- softcapValue: 0 (disabled) or positive number (e.g. 15)
- lr (1e-5 to 1e-2): learning rate
- weightDecay (0 to 0.5): AdamW weight decay
- warmupRatio (0 to 0.5): fraction of training for LR warmup
- cooldownRatio (0 to 0.5): fraction of training for LR cooldown
- batchSize (4-32): batch size
- seqLen (64-256): sequence length
- trainSeconds (30-90): wall-clock training budget

Constraints:
- Total params should stay under ~3M (WebGPU memory limits)
- Training runs for trainSeconds wall-clock, so bigger models = fewer steps
- The dataset is ~1MB of Shakespeare text (byte-level)
- Each experiment takes 30-90 seconds real time

Strategy tips:
- Start with small changes to understand what matters
- Learning rate is usually the highest-leverage knob
- Wider models (bigger nEmbd) often beat deeper ones at this scale
- relu² tends to work well for small models
- Don't change too many things at once`;
}

export function buildUserPrompt(
	history: ExperimentRecord[],
	bestConfig: ExperimentConfig,
	bestBpb: number
): string {
	let msg = `Current best config (val_bpb = ${bestBpb.toFixed(4)}):\n`;
	msg += '```json\n' + JSON.stringify(bestConfig, null, 2) + '\n```\n\n';

	if (history.length > 0) {
		msg += 'Experiment history (most recent first):\n';
		const recent = history.slice(-10).reverse();
		for (const exp of recent) {
			const badge = exp.kept ? 'KEPT' : 'DISCARDED';
			msg += `\n#${exp.id} [${badge}] val_bpb=${exp.valBpb.toFixed(4)} (${exp.totalSteps} steps, ${(exp.elapsed / 1000).toFixed(1)}s)\n`;
			msg += `  Reasoning: ${exp.reasoning}\n`;
			const diff = configDiff(bestConfig, exp.config);
			if (diff) msg += `  Changes: ${diff}\n`;
		}
	}

	msg += `\nPropose the next experiment. Respond with ONLY a JSON object containing:
{
  "reasoning": "one sentence explaining your hypothesis",
  "config": { ... full ExperimentConfig ... }
}`;

	return msg;
}

function configDiff(a: ExperimentConfig, b: ExperimentConfig): string {
	const diffs: string[] = [];
	for (const key of Object.keys(a) as (keyof ExperimentConfig)[]) {
		if (a[key] !== b[key]) {
			diffs.push(`${key}: ${a[key]} → ${b[key]}`);
		}
	}
	return diffs.join(', ');
}

export { configDiff };
