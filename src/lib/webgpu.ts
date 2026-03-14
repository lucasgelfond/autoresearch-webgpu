import { init, defaultDevice } from '@jax-js/jax';

export type WebGPUStatus =
	| { ok: true; adapterInfo: GPUAdapterInfo }
	| { ok: false; reason: string };

export async function initWebGPU(): Promise<WebGPUStatus> {
	if (!navigator.gpu) {
		return { ok: false, reason: 'WebGPU is not supported in this browser.' };
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		return { ok: false, reason: 'No WebGPU adapter found.' };
	}

	const adapterInfo = adapter.info;

	try {
		await init('webgpu');
		defaultDevice('webgpu');
	} catch (e) {
		const msg = e instanceof Error ? e.message : String(e);
		return { ok: false, reason: `Failed to initialize jax-js: ${msg}` };
	}

	return { ok: true, adapterInfo };
}
