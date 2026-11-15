import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fr } from './fr';
import { t, locale, htmlLang } from './index';

type Leaf = { path: string; value: string };
type Builder = { path: string; value: (...args: never[]) => string };

function collect(node: unknown, path = ''): Leaf[] {
	if (typeof node === 'string') return [{ path, value: node }];
	if (node && typeof node === 'object') {
		return Object.entries(node as Record<string, unknown>).flatMap(([key, child]) =>
			collect(child, path ? `${path}.${key}` : key)
		);
	}
	return [];
}

function collectBuilders(node: unknown, path = ''): Builder[] {
	if (typeof node === 'function') return [{ path, value: node as Builder['value'] }];
	if (node && typeof node === 'object') {
		return Object.entries(node as Record<string, unknown>).flatMap(([key, child]) =>
			collectBuilders(child, path ? `${path}.${key}` : key)
		);
	}
	return [];
}

const strings = collect(fr);
const builders = collectBuilders(fr);

test('the active locale is French', () => {
	assert.equal(locale, 'fr');
	assert.equal(htmlLang, 'fr');
	assert.equal(t, fr);
});

test('the string table is not empty', () => {
	assert.ok(strings.length > 50, `expected many strings, got ${strings.length}`);
	assert.ok(builders.length > 5, `expected dynamic messages, got ${builders.length}`);
});

test('no message is empty or padded', () => {
	for (const { path, value } of strings) {
		assert.ok(value.trim().length > 0, `${path} is empty`);
		// `header.credits` holds sentence fragments glued around links, so the
		// spaces there are part of the copy.
		if (!path.startsWith('header.credits')) {
			assert.equal(value.trim(), value, `${path} has surrounding whitespace`);
		}
		assert.ok(!/ {2,}/.test(value.replace(/\u00a0/g, ' ')), `${path} has repeated spaces`);
	}
});

test('messages use plain ASCII apostrophes and quotes', () => {
	for (const { path, value } of strings) {
		assert.ok(!/[\u2018\u2019\u201b]/.test(value), `${path} uses a curly apostrophe`);
		assert.ok(!/[\u201c\u201d]/.test(value), `${path} uses a curly quote`);
	}
});

test('French typography: no-break space before ? ! : ;', () => {
	for (const { path, value } of strings) {
		assert.ok(!/[^\s\u00a0][?!:;]/.test(value), `${path} is missing a no-break space before punctuation`);
	}
});

test('run status messages are in French and keep the technical labels', () => {
	assert.match(fr.status.step(4, '1.2345', '0.5'), /^étape 4 \| perte 1\.2345 \| 0\.5s$/);
	assert.match(fr.status.done('1.2345', 12), /^terminé — val_bpb\u00a0: 1\.2345 \| 12 étapes$/);
	assert.match(fr.status.experiment('essai'), /^expérience\u00a0: essai$/);
	assert.match(fr.status.experimentDone(3, true, '1.5000'), /^#3 retenue/);
	assert.match(fr.status.experimentDone(3, false, '1.5000'), /^#3 écartée/);
	assert.match(fr.status.thinking('réfléchir'), /^réflexion\u00a0: réfléchir$/);
	assert.match(fr.status.error('boum'), /^erreur\u00a0: boum$/);
});

test('error messages are in French and keep the details', () => {
	assert.match(fr.errors.jaxInit('GPU perdu'), /GPU perdu$/);
	assert.match(fr.errors.timeout(30), /30\u00a0s/);
	assert.match(fr.errors.api(500, 'boom'), /^API 500\u00a0: boom$/);
	assert.match(fr.errors.fetchFailed(new Error('réseau')), /réseau$/);
});
