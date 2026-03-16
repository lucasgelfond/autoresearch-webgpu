export type ResearchProvider = 'anthropic' | 'openai';

export type ResearchEndpointProfile = {
	id: string;
	name: string;
	provider: ResearchProvider;
	baseUrl: string;
	apiKey: string;
	model: string;
};

export const DEFAULT_ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';
export const DEFAULT_OPENAI_URL = 'https://api.openai.com/v1/chat/completions';

function profileId(): string {
	if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
		return crypto.randomUUID();
	}
	return `profile-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function defaultBaseUrl(provider: ResearchProvider): string {
	return provider === 'anthropic' ? DEFAULT_ANTHROPIC_URL : DEFAULT_OPENAI_URL;
}

export function createResearchEndpointProfile(
	provider: ResearchProvider = 'anthropic'
): ResearchEndpointProfile {
	return {
		id: profileId(),
		name: provider === 'anthropic' ? 'Anthropic' : 'OpenAI Compatible',
		provider,
		baseUrl: defaultBaseUrl(provider),
		apiKey: '',
		model: provider === 'anthropic' ? 'claude-sonnet-4-6' : '',
	};
}

export function normalizeProfile(profile: ResearchEndpointProfile): ResearchEndpointProfile {
	return {
		...profile,
		name: profile.name.trim(),
		baseUrl: profile.baseUrl.trim(),
		apiKey: profile.apiKey.trim(),
		model: profile.model.trim(),
	};
}

export function isConfiguredProfile(profile: ResearchEndpointProfile | null | undefined): boolean {
	if (!profile) return false;
	const normalized = normalizeProfile(profile);
	return Boolean(normalized.baseUrl && normalized.apiKey && normalized.model);
}
