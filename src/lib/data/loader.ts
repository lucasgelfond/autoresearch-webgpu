import { numpy as np } from '@jax-js/jax';

export type Batch = {
	input: np.Array;  // [batchSize, seqLen] int32
	target: np.Array; // [batchSize, seqLen] int32
};

export class DataLoader {
	private data: Uint8Array;
	private pos: number = 0;

	constructor(data: Uint8Array) {
		this.data = data;
	}

	static async fetch(url: string): Promise<DataLoader> {
		const response = await globalThis.fetch(url);
		const buffer = await response.arrayBuffer();
		return new DataLoader(new Uint8Array(buffer));
	}

	get length(): number {
		return this.data.length;
	}

	nextBatch(batchSize: number, seqLen: number): Batch {
		const input = new Int32Array(batchSize * seqLen);
		const target = new Int32Array(batchSize * seqLen);

		for (let b = 0; b < batchSize; b++) {
			// Wrap around if we're near the end
			if (this.pos + seqLen + 1 > this.data.length) {
				this.pos = 0;
			}
			for (let t = 0; t < seqLen; t++) {
				input[b * seqLen + t] = this.data[this.pos + t];
				target[b * seqLen + t] = this.data[this.pos + t + 1];
			}
			this.pos += seqLen;
		}

		return {
			input: np.array(input, { dtype: np.int32 }).reshape([batchSize, seqLen]),
			target: np.array(target, { dtype: np.int32 }).reshape([batchSize, seqLen])
		};
	}

	reset(): void {
		this.pos = 0;
	}
}
