import { BASELINE_CODE } from '../baseline';
import type { GenerateOptions, ResearchProvider } from './types';

const RESPONSE = JSON.stringify({
	reasoning: 'Mock provider: rerunning the baseline.',
	code: BASELINE_CODE
});

export class MockProvider implements ResearchProvider {
	async generate({ onToken }: GenerateOptions): Promise<string> {
		onToken?.(RESPONSE);
		return RESPONSE;
	}
}
