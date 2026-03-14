import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, platform }) => {
	const apiKey = platform?.env?.ANTHROPIC_API_KEY;
	if (!apiKey) {
		return json({ error: 'API key not configured' }, { status: 500 });
	}

	const rateLimiter = platform?.env?.RATE_LIMITER;
	if (rateLimiter) {
		const { success } = await rateLimiter.limit({ key: 'global' });
		if (!success) {
			return json({ error: 'Rate limited. Max 1 request every 10 seconds.' }, { status: 429 });
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

	try {
		const parsed = JSON.parse(text);
		if (parsed.code && parsed.reasoning) return json(parsed);
	} catch {}

	const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
	if (fenceMatch) {
		try {
			const parsed = JSON.parse(fenceMatch[1].trim());
			if (parsed.code) return json(parsed);
		} catch {}
	}

	try {
		const start = text.indexOf('{');
		if (start >= 0) {
			let depth = 0, end = start;
			for (let i = start; i < text.length; i++) {
				if (text[i] === '{') depth++;
				else if (text[i] === '}') { depth--; if (depth === 0) { end = i; break; } }
			}
			const parsed = JSON.parse(text.slice(start, end + 1));
			if (parsed.code) return json(parsed);
		}
	} catch {}

	return json({ error: 'Could not parse response', raw: text.slice(0, 500) }, { status: 422 });
};
