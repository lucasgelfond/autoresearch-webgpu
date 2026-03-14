/**
 * Tokenize a text file into a binary blob for browser training.
 *
 * Usage: npx tsx scripts/prepare-data.ts <input.txt> [output.bin]
 *
 * Output is a raw Uint8Array — each byte is already a token ID.
 * The browser fetches this and slices it into (input, target) pairs.
 */

import { readFileSync, writeFileSync } from 'fs';

const inputPath = process.argv[2];
if (!inputPath) {
	console.error('Usage: npx tsx scripts/prepare-data.ts <input.txt> [output.bin]');
	process.exit(1);
}

const outputPath = process.argv[3] ?? 'static/data/train.bin';

const text = readFileSync(inputPath, 'utf-8');
const bytes = new TextEncoder().encode(text);

// Split 90/10 train/val
const splitIdx = Math.floor(bytes.length * 0.9);
const trainBytes = bytes.slice(0, splitIdx);
const valBytes = bytes.slice(splitIdx);

writeFileSync(outputPath, trainBytes);
writeFileSync(outputPath.replace('train.bin', 'val.bin'), valBytes);

console.log(`Train: ${trainBytes.length} bytes → ${outputPath}`);
console.log(`Val:   ${valBytes.length} bytes → ${outputPath.replace('train.bin', 'val.bin')}`);
console.log(`Vocab: 256 (byte-level)`);
