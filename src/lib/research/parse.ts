/**
 * Parse a model response into { code, reasoning }.
 */
export function parseResponse(text: string): { code: string; reasoning: string } | null {
	try {
		const parsed = JSON.parse(text);
		if (parsed.code && parsed.reasoning) return parsed;
	} catch {}

	const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
	if (fenceMatch) {
		try {
			const parsed = JSON.parse(fenceMatch[1].trim());
			if (parsed.code) return { code: parsed.code, reasoning: parsed.reasoning || '' };
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
			if (parsed.code) return { code: parsed.code, reasoning: parsed.reasoning || '' };
		}
	} catch {}

	// Small models often emit slightly invalid JSON (raw newlines, bad escapes) around the
	// code string; accept a leniently extracted program as long as it is complete.
	const partial = extractPartial(text);
	if (partial.code?.includes('onReturn')) return { code: partial.code, reasoning: partial.reasoning ?? '' };

	return null;
}

/** Extract code and reasoning from partial or malformed JSON, e.g. as it streams in. */
export function extractPartial(partial: string): { code?: string; reasoning?: string } {
	const result: { code?: string; reasoning?: string } = {};

	// Try to find reasoning field
	const reasoningMatch = partial.match(/"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"/);
	if (reasoningMatch) {
		try { result.reasoning = JSON.parse('"' + reasoningMatch[1] + '"'); } catch {}
	}

	// Try to find code field — it may be incomplete
	const codeStart = partial.indexOf('"code"');
	if (codeStart === -1) return result;

	// Find the opening quote of the code value
	const colonAfterCode = partial.indexOf(':', codeStart + 6);
	if (colonAfterCode === -1) return result;

	const quoteStart = partial.indexOf('"', colonAfterCode);
	if (quoteStart === -1) return result;

	// Extract the code value, handling escaped characters
	// Walk through the string handling escapes
	let code = '';
	let i = quoteStart + 1;
	while (i < partial.length) {
		if (partial[i] === '\\' && i + 1 < partial.length) {
			const next = partial[i + 1];
			if (next === '"') { code += '"'; i += 2; }
			else if (next === '\\') { code += '\\'; i += 2; }
			else if (next === 'n') { code += '\n'; i += 2; }
			else if (next === 't') { code += '\t'; i += 2; }
			else if (next === 'r') { code += '\r'; i += 2; }
			else { code += partial[i]; i++; }
		} else if (partial[i] === '"') {
			// End of string
			break;
		} else {
			code += partial[i];
			i++;
		}
	}

	if (code.length > 0) {
		result.code = code;
	}

	return result;
}
