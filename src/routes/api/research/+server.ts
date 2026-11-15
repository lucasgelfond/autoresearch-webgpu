import { json } from '@sveltejs/kit';
import { dev } from '$app/environment';
import type { RequestHandler } from './$types';
import { parseClaudeResponse } from '$lib/research/parse';
import { t } from '$lib/i18n';

const ALLOWED_ORIGIN = 'https://autoresearch.lucasgelfond.online';

export const POST: RequestHandler = async ({ request, platform }) => {
	if (!dev) {
		const origin = request.headers.get('origin');
		if (origin !== ALLOWED_ORIGIN) {
			return json({ error: t.errors.server.forbidden }, { status: 403 });
		}
	}

	const apiKey = platform?.env?.ANTHROPIC_API_KEY;
	if (!apiKey) {
		return json({ error: t.errors.server.apiKeyMissing }, { status: 500 });
	}

	const rateLimiter = platform?.env?.RATE_LIMITER;
	if (rateLimiter) {
		const { success } = await rateLimiter.limit({ key: 'global' });
		if (!success) {
			return json({ error: t.errors.server.rateLimited }, { status: 429 });
		}
	}

	const { systemPrompt, userPrompt, stream } = await request.json();

	const body: Record<string, unknown> = {
		model: 'claude-sonnet-4-6',
		max_tokens: 8192,
		system: [
			{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }
		],
		messages: [{ role: 'user', content: userPrompt }],
	};

	if (stream) {
		body.stream = true;

		const response = await fetch('https://api.anthropic.com/v1/messages', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'x-api-key': apiKey,
				'anthropic-version': '2023-06-01',
			},
			body: JSON.stringify(body),
		});

		if (!response.ok) {
			const error = await response.text();
			return json({ error }, { status: response.status });
		}

		// Pass through the SSE stream from Anthropic
		return new Response(response.body, {
			headers: {
				'Content-Type': 'text/event-stream',
				'Cache-Control': 'no-cache',
				'Connection': 'keep-alive',
			},
		});
	}

	// Non-streaming path (kept for backwards compat)
	const response = await fetch('https://api.anthropic.com/v1/messages', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'x-api-key': apiKey,
			'anthropic-version': '2023-06-01',
		},
		body: JSON.stringify(body),
	});

	if (!response.ok) {
		const error = await response.text();
		return json({ error }, { status: response.status });
	}

	const data = await response.json();
	const text = data.content[0].text;

	const parsed = parseClaudeResponse(text);
	if (parsed) return json(parsed);

	return json({ error: t.errors.server.cannotParse, raw: text.slice(0, 500) }, { status: 422 });
};
