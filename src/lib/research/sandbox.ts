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
	let resolved = false;

	const onStep = (m: StepMetrics) => {
		lastStep = m.step;
		lastElapsed = m.elapsed;
		callbacks.onStep(m);
	};

	let resolveResult: ((result: RunResult) => void) | null = null;
	const resultPromise = new Promise<RunResult>((resolve) => {
		resolveResult = resolve;
	});

	const onReturn = (r: TrainResult & { valBpb?: number }) => {
		resolved = true;
		resolveResult?.({
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

	const runCallbacks: {
		trainData: DataLoader;
		valData: DataLoader;
		trainSeconds: number;
		signal: AbortSignal;
		onStep: typeof onStep;
		onReturn: (r: TrainResult & { valBpb?: number }) => void;
	} = {
		trainData,
		valData,
		trainSeconds,
		signal: callbacks.signal,
		onStep,
		onReturn,
	};

	let timeoutId: ReturnType<typeof setTimeout> | null = null;
	const timeoutPromise = new Promise<never>((_, reject) => {
		// Timeout: 2x the training budget (minimum 30s for model init/inference rebuild)
		const timeoutMs = Math.max(trainSeconds * 2000, 30000);
		timeoutId = setTimeout(() => {
			reject(new Error(`Training exceeded ${timeoutMs / 1000}s timeout`));
		});
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

		const runPromise = (async () => {
			trainData.reset();
			await fn(prepare, runCallbacks);
			if (!resolved) {
				throw new Error(
					callbacks.signal.aborted
						? 'Training aborted before returning a result'
						: 'Training completed without calling onReturn'
				);
			}
			return resultPromise;
		})();

		return await Promise.race([runPromise, resultPromise, timeoutPromise]);
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
		if (timeoutId !== null) {
			clearTimeout(timeoutId);
		}
	}
}
