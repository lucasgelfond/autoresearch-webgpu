import { json } from '@sveltejs/kit';
import { dev } from '$app/environment';
import type { RequestHandler } from './$types';
import { parseClaudeResponse } from '$lib/research/parse';
import {
	DEFAULT_ANTHROPIC_URL,
	type ResearchEndpointProfile,
	type ResearchProvider,
} from '$lib/research/providers';

const ALLOWED_ORIGIN = 'https://autoresearch.lucasgelfond.online';

type ResearchRequestBody = {
	systemPrompt: string;
	userPrompt: string;
	stream?: boolean;
	profile?: ResearchEndpointProfile | null;
};

type ResolvedProfile = {
	provider: ResearchProvider;
	baseUrl: string;
	apiKey: string;
	model: string;
};

function textContent(value: unknown): string {
	if (typeof value === 'string') return value;
	if (Array.isArray(value)) {
		return value
			.map((item) => {
				if (typeof item === 'string') return item;
				if (item && typeof item === 'object' && 'text' in item && typeof item.text === 'string') {
					return item.text;
				}
				return '';
			})
			.join('');
	}
	return '';
}

function resolveProfile(
	requestProfile: ResearchEndpointProfile | null | undefined,
	platform: App.Platform | undefined
): ResolvedProfile | null {
	if (
		requestProfile &&
		requestProfile.baseUrl.trim() &&
		requestProfile.apiKey.trim() &&
		requestProfile.model.trim()
	) {
		return {
			provider: requestProfile.provider === 'openai' ? 'openai' : 'anthropic',
			baseUrl: requestProfile.baseUrl.trim(),
			apiKey: requestProfile.apiKey.trim(),
			model: requestProfile.model.trim(),
		};
	}

	const apiKey = platform?.env?.ANTHROPIC_API_KEY;
	if (!apiKey) return null;

	return {
		provider: 'anthropic',
		baseUrl: DEFAULT_ANTHROPIC_URL,
		apiKey,
		model: 'claude-sonnet-4-6',
	};
}

function buildUpstreamRequest(
	profile: ResolvedProfile,
	systemPrompt: string,
	userPrompt: string,
	stream: boolean
): { url: string; init: RequestInit } {
	if (profile.provider === 'openai') {
		return {
			url: profile.baseUrl,
			init: {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${profile.apiKey}`,
				},
				body: JSON.stringify({
					model: profile.model,
					stream,
					temperature: 1,
					max_tokens: 8192,
					messages: [
						{ role: 'system', content: systemPrompt },
						{ role: 'user', content: userPrompt },
					],
				}),
			},
		};
	}

	return {
		url: profile.baseUrl,
		init: {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'x-api-key': profile.apiKey,
				'anthropic-version': '2023-06-01',
			},
			body: JSON.stringify({
				model: profile.model,
				max_tokens: 8192,
				stream,
				system: [
					{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }
				],
				messages: [{ role: 'user', content: userPrompt }],
			}),
		},
	};
}

function extractStreamingText(event: any, provider: ResearchProvider): string {
	if (provider === 'openai') {
		const choice = event?.choices?.[0];
		return textContent(choice?.delta?.content);
	}

	if (event?.type === 'content_block_delta') {
		return textContent(event?.delta?.text);
	}

	return '';
}

function extractResponseText(data: any, provider: ResearchProvider): string {
	if (provider === 'openai') {
		return textContent(data?.choices?.[0]?.message?.content);
	}

	return textContent(data?.content);
}

function normalizedSseResponse(upstream: Response, provider: ResearchProvider): Response {
	const encoder = new TextEncoder();
	const decoder = new TextDecoder();

	const stream = new ReadableStream<Uint8Array>({
		async start(controller) {
			if (!upstream.body) {
				controller.error(new Error('Upstream response body missing'));
				return;
			}

			const reader = upstream.body.getReader();
			let buffer = '';

			try {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;

					buffer += decoder.decode(value, { stream: true });
					const lines = buffer.split('\n');
					buffer = lines.pop() || '';

					for (const rawLine of lines) {
						const line = rawLine.trim();
						if (!line.startsWith('data:')) continue;

						const payload = line.slice(5).trim();
						if (!payload || payload === '[DONE]') continue;

						try {
							const event = JSON.parse(payload);
							const text = extractStreamingText(event, provider);
							if (!text) continue;
							controller.enqueue(
								encoder.encode(`data: ${JSON.stringify({ type: 'text_delta', text })}\n\n`)
							);
						} catch {
							// Ignore malformed upstream chunks and continue streaming.
						}
					}
				}

				controller.enqueue(encoder.encode('data: [DONE]\n\n'));
				controller.close();
			} catch (error) {
				controller.error(error);
			} finally {
				reader.releaseLock();
			}
		},
	});

	return new Response(stream, {
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			Connection: 'keep-alive',
		},
	});
}

export const POST: RequestHandler = async ({ request, platform }) => {
	if (!dev) {
		const origin = request.headers.get('origin');
		if (origin !== ALLOWED_ORIGIN) {
			return json({ error: 'Forbidden' }, { status: 403 });
		}
	}

	const rateLimiter = platform?.env?.RATE_LIMITER;
	if (rateLimiter) {
		const { success } = await rateLimiter.limit({ key: 'global' });
		if (!success) {
			return json({ error: 'Rate limited. Max 1 request every 10 seconds.' }, { status: 429 });
		}
	}

	const { systemPrompt, userPrompt, stream = false, profile } = await request.json() as ResearchRequestBody;
	const resolvedProfile = resolveProfile(profile, platform);
	if (!resolvedProfile) {
		return json({
			error: 'No research backend configured. Add a profile in the UI or set ANTHROPIC_API_KEY for the server.'
		}, { status: 500 });
	}

	const upstream = buildUpstreamRequest(resolvedProfile, systemPrompt, userPrompt, stream);
	const response = await fetch(upstream.url, upstream.init);

	if (!response.ok) {
		const error = await response.text();
		return json({ error }, { status: response.status });
	}

	if (stream) {
		return normalizedSseResponse(response, resolvedProfile.provider);
	}

	const data = await response.json();
	const text = extractResponseText(data, resolvedProfile.provider);
	const parsed = parseClaudeResponse(text);
	if (parsed) return json(parsed);

	return json({ error: 'Could not parse response', raw: text.slice(0, 500) }, { status: 422 });
};
