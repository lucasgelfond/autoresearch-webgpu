import { json } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';

// Simple in-memory rate limiter: max 10 requests per IP per minute
const requests = new Map<string, number[]>();
const MAX_REQUESTS = 10;
const WINDOW_MS = 60_000;

function rateLimit(ip: string): boolean {
	const now = Date.now();
	const timestamps = requests.get(ip) ?? [];
	const recent = timestamps.filter((t) => now - t < WINDOW_MS);

	if (recent.length >= MAX_REQUESTS) {
		requests.set(ip, recent);
		return false;
	}

	recent.push(now);
	requests.set(ip, recent);
	return true;
}

// Clean up stale entries every 5 minutes
setInterval(() => {
	const now = Date.now();
	for (const [ip, timestamps] of requests) {
		const recent = timestamps.filter((t) => now - t < WINDOW_MS);
		if (recent.length === 0) requests.delete(ip);
		else requests.set(ip, recent);
	}
}, 300_000);

export const POST = async ({ request, getClientAddress }: { request: Request; getClientAddress: () => string }) => {
	const ip = getClientAddress();
	if (!rateLimit(ip)) {
		return json({ error: 'Rate limited. Max 10 requests per minute.' }, { status: 429 });
	}

	const { systemPrompt, userPrompt } = await request.json();

	const response = await fetch('https://api.anthropic.com/v1/messages', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'x-api-key': env.ANTHROPIC_API_KEY ?? '',
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
