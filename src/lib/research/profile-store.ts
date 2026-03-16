import {
	createResearchEndpointProfile,
	normalizeProfile,
	type ResearchEndpointProfile,
} from './providers';

const STORAGE_KEY = 'autoresearch.research-profiles.v1';
const ACTIVE_KEY = 'autoresearch.active-research-profile.v1';

function canUseStorage(): boolean {
	return typeof window !== 'undefined' && typeof localStorage !== 'undefined';
}

export function loadResearchProfiles(): ResearchEndpointProfile[] {
	if (!canUseStorage()) return [createResearchEndpointProfile()];

	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (!raw) return [createResearchEndpointProfile()];

		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed) || parsed.length === 0) {
			return [createResearchEndpointProfile()];
		}

		return parsed
			.filter((item): item is ResearchEndpointProfile => Boolean(item && typeof item === 'object'))
			.map((item) => normalizeProfile({
				id: String(item.id ?? createResearchEndpointProfile().id),
				name: String(item.name ?? ''),
				provider: item.provider === 'openai' ? 'openai' : 'anthropic',
				baseUrl: String(item.baseUrl ?? ''),
				apiKey: String(item.apiKey ?? ''),
				model: String(item.model ?? ''),
			}));
	} catch {
		return [createResearchEndpointProfile()];
	}
}

export function saveResearchProfiles(profiles: ResearchEndpointProfile[]): void {
	if (!canUseStorage()) return;
	localStorage.setItem(STORAGE_KEY, JSON.stringify(profiles.map(normalizeProfile)));
}

export function loadActiveResearchProfileId(profiles: ResearchEndpointProfile[]): string | null {
	if (profiles.length === 0) return null;
	if (!canUseStorage()) return profiles[0].id;

	const active = localStorage.getItem(ACTIVE_KEY);
	if (active && profiles.some((profile) => profile.id === active)) {
		return active;
	}

	return profiles[0].id;
}

export function saveActiveResearchProfileId(profileId: string | null): void {
	if (!canUseStorage()) return;
	if (profileId) localStorage.setItem(ACTIVE_KEY, profileId);
	else localStorage.removeItem(ACTIVE_KEY);
}
