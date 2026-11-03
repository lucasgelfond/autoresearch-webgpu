/**
 * sandbox.ts — Executes generated training code with injected globals.
 */

import { getPrepareGlobals, type Params, type ForwardFn, type StepMetrics, type TrainResult } from '../prepare';
import { DataLoader } from '../data/loader';

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

	const onStep = (m: StepMetrics) => {
		lastStep = m.step;
		lastElapsed = m.elapsed;
		callbacks.onStep(m);
	};

	const { promise, resolve } = Promise.withResolvers<RunResult>();

	const onReturn = (r: TrainResult & { valBpb?: number }) => {
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

	// Timeout: 2x the training budget (minimum 30s for model init/inference rebuild).
	// Races the run so hung code or a missing onReturn fails the experiment instead of the loop.
	const timeoutMs = Math.max(trainSeconds * 2000, 30000);
	const timeoutAbort = new AbortController();
	let timer: ReturnType<typeof setTimeout> | undefined;
	const timedOut = new Promise<never>((_, reject) => {
		timer = setTimeout(() => {
			timeoutAbort.abort();
			reject(new Error(`Training exceeded ${timeoutMs / 1000}s timeout`));
		}, timeoutMs);
	});

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
		await Promise.race([
			fn(prepare, {
				trainData,
				valData,
				trainSeconds,
				signal: AbortSignal.any([callbacks.signal, timeoutAbort.signal]),
				onStep,
				onReturn,
			}),
			timedOut
		]);
		return await Promise.race([promise, timedOut]);
	} catch (e) {
		const msg = e instanceof Error ? e.message : String(e);
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
	} finally {
		clearTimeout(timer);
	}
}
