import { numpy as np } from '@jax-js/jax';
import { opfs } from '@jax-js/loaders';
import type { Params } from './model/gpt';

type ParamMeta = { key: string; shape: number[]; dtype: string; offset: number; byteLength: number };

export async function saveWeights(experimentId: number, params: Params): Promise<string> {
	const path = `weights/exp-${experimentId}.bin`;
	const metaPath = `weights/exp-${experimentId}.meta.json`;

	const metas: ParamMeta[] = [];
	const buffers: Uint8Array[] = [];
	let totalBytes = 0;

	// Single pass: extract data and compute layout
	for (const key of Object.keys(params)) {
		const arr = params[key];
		const data = await arr.ref.data();
		const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		metas.push({
			key,
			shape: arr.ref.shape,
			dtype: arr.ref.dtype,
			offset: totalBytes,
			byteLength: data.byteLength
		});
		buffers.push(bytes);
		totalBytes += data.byteLength;
	}

	// Pack into single buffer
	const buffer = new Uint8Array(totalBytes);
	for (let i = 0; i < buffers.length; i++) {
		buffer.set(buffers[i], metas[i].offset);
	}

	await opfs.write(path, buffer);
	await opfs.write(metaPath, new TextEncoder().encode(JSON.stringify(metas)));

	return path;
}

export async function loadWeights(experimentId: number): Promise<Params | null> {
	const path = `weights/exp-${experimentId}.bin`;
	const metaPath = `weights/exp-${experimentId}.meta.json`;

	const metaBytes = await opfs.read(metaPath);
	if (!metaBytes) return null;

	const metas: ParamMeta[] = JSON.parse(new TextDecoder().decode(metaBytes));

	const buffer = await opfs.read(path);
	if (!buffer) return null;

	const params: Params = {};
	for (const meta of metas) {
		const slice = buffer.slice(meta.offset, meta.offset + meta.byteLength);
		let typedArray: Float32Array | Int32Array | Uint32Array;

		switch (meta.dtype) {
			case 'float32':
				typedArray = new Float32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
				break;
			case 'int32':
				typedArray = new Int32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
				break;
			case 'uint32':
				typedArray = new Uint32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
				break;
			default:
				typedArray = new Float32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4);
		}

		params[meta.key] = np.array(typedArray as any, { shape: meta.shape, dtype: meta.dtype as any });
	}

	return params;
}

export async function deleteWeights(experimentId: number): Promise<void> {
	await opfs.remove(`weights/exp-${experimentId}.bin`).catch(() => {});
	await opfs.remove(`weights/exp-${experimentId}.meta.json`).catch(() => {});
}
