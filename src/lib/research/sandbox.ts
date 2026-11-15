/**
 * sandbox.ts — Executes generated training code with injected globals.
 */

import { getPrepareGlobals, type Params, type ForwardFn, type StepMetrics, type TrainResult } from '../prepare';
import { DataLoader } from '../data/loader';
import { t } from '../i18n';

export type RunResult = {
	valBpb: number;
	totalSteps: number;
	elapsed: number;
	params: Params;
	forward: ForwardFn;
	vocabSize: number;
	batchSize: number;
	seqLen: number;
	error?: string;
};

export async function executeTrainCode(
	code: string,
	trainData: DataLoader,
	valData: DataLoader,
	trainSeconds: number,
	callbacks: {
		onStep: (m: StepMetrics) => void;
		signal: AbortSignal;
	}
): Promise<RunResult> {
	const prepare = getPrepareGlobals();

	let lastStep = 0;
	let lastElapsed = 0;
	let resolved = false;

	const onStep = (m: StepMetrics) => {
		lastStep = m.step;
		lastElapsed = m.elapsed;
		callbacks.onStep(m);
	};

	const { promise, resolve, reject } = Promise.withResolvers<RunResult>();

	const onReturn = (r: TrainResult & { valBpb?: number }) => {
		resolved = true;
		resolve({
			valBpb: r.valBpb ?? Infinity,
			totalSteps: lastStep,
			elapsed: lastElapsed,
			params: r.params,
			forward: r.forward,
			vocabSize: r.vocabSize,
			batchSize: r.batchSize,
			seqLen: r.seqLen,
		});
	};

	// Timeout: 2x the training budget (minimum 30s for model init/inference rebuild)
	const timeoutMs = Math.max(trainSeconds * 2000, 30000);
	const timeout = setTimeout(() => {
		if (!resolved) reject(new Error(t.errors.timeout(timeoutMs / 1000)));
	}, timeoutMs);

	try {
		// Build the function body with all globals destructured
		const globalNames = Object.keys(prepare);
		const callbackNames = ['trainData', 'valData', 'trainSeconds', 'signal', 'onStep', 'onReturn'];

		const fn = new Function(
			'__prepare__',
			'__callbacks__',
			`
			const { ${globalNames.join(', ')} } = __prepare__;
			const { ${callbackNames.join(', ')} } = __callbacks__;
			return (async () => {
				${code}
			})();
			`
		);

		trainData.reset();
		await fn(prepare, {
			trainData,
			valData,
			trainSeconds,
			signal: callbacks.signal,
			onStep,
			onReturn,
		});
	} catch (e) {
		clearTimeout(timeout);
		const msg = e instanceof Error ? e.message : String(e);
		// If onReturn was never called, resolve with error
		return {
			valBpb: Infinity,
			totalSteps: lastStep,
			elapsed: lastElapsed,
			params: {},
			forward: () => { throw new Error('no model'); },
			vocabSize: 256,
			batchSize: 8,
			seqLen: 128,
			error: msg,
		};
	}
	clearTimeout(timeout);

	return promise;
}
