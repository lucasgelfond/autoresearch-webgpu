import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, platform }) => {
	const apiKey = platform?.env?.ANTHROPIC_API_KEY;
	if (!apiKey) {
		return json({ error: 'API key not configured' }, { status: 500 });
	}

	// Cloudflare rate limiting
	const rateLimiter = platform?.env?.RATE_LIMITER;
	if (rateLimiter) {
		const { success } = await rateLimiter.limit({ key: 'global' });
		if (!success) {
			return json({ error: 'Rate limited. Max 1 request every 10 seconds.' }, { status: 429 });
		}
	}

	const { systemPrompt, userPrompt } = await request.json();

	const response = await fetch('https://api.anthropic.com/v1/messages', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'x-api-key': apiKey,
			'anthropic-version': '2023-06-01'
		},
		body: JSON.stringify({
			model: 'claude-sonnet-4-6',
			max_tokens: 1024,
			system: systemPrompt,
			messages: [{ role: 'user', content: userPrompt }]
		})
	});

	if (!response.ok) {
		const error = await response.text();
		return json({ error }, { status: response.status });
	}

	const data = await response.json();
	const text = data.content[0].text;

	const jsonMatch = text.match(/\{[\s\S]*\}/);
	if (!jsonMatch) {
		return json({ error: 'No JSON found in response', raw: text }, { status: 422 });
	}

	try {
		const parsed = JSON.parse(jsonMatch[0]);
		return json(parsed);
	} catch {
		return json({ error: 'Invalid JSON in response', raw: text }, { status: 422 });
	}
};
