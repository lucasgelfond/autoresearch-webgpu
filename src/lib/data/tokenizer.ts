const encoder = new TextEncoder();
const decoder = new TextDecoder();

export const VOCAB_SIZE = 256;

export function encode(text: string): Uint8Array {
	return encoder.encode(text);
}

export function decode(ids: ArrayLike<number>): string {
	return decoder.decode(new Uint8Array(ids));
}
