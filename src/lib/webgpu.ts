import { init, defaultDevice } from '@jax-js/jax';
import { t } from './i18n';

export type WebGPUStatus =
	| { ok: true; adapterInfo: GPUAdapterInfo }
	| { ok: false; reason: string };

export async function initWebGPU(): Promise<WebGPUStatus> {
	if (!navigator.gpu) {
		return { ok: false, reason: t.errors.webgpuUnsupported };
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		return { ok: false, reason: t.errors.noAdapter };
	}

	const adapterInfo = adapter.info;

	try {
		await init('webgpu');
		defaultDevice('webgpu');
	} catch (e) {
		const msg = e instanceof Error ? e.message : String(e);
		return { ok: false, reason: t.errors.jaxInit(msg) };
	}

	return { ok: true, adapterInfo };
}
