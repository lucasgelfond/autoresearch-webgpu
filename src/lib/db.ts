import { PGlite } from '@electric-sql/pglite';
import { clearSavedWeights } from './weights';

let db: PGlite | null = null;

const SCHEMA = `
	CREATE TABLE IF NOT EXISTS experiments (
		id SERIAL PRIMARY KEY,
		name TEXT NOT NULL DEFAULT '',
		source TEXT NOT NULL DEFAULT 'manual',
		code TEXT NOT NULL,
		reasoning TEXT NOT NULL DEFAULT '',
		val_bpb REAL NOT NULL,
		elapsed REAL NOT NULL,
		total_steps INTEGER NOT NULL,
		kept BOOLEAN NOT NULL DEFAULT false,
		error TEXT,
		loss_curve JSONB,
		weights_path TEXT,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);

	CREATE TABLE IF NOT EXISTS loss_steps (
		id SERIAL PRIMARY KEY,
		experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
		step INTEGER NOT NULL,
		loss REAL NOT NULL
	);

	CREATE INDEX IF NOT EXISTS idx_loss_steps_exp ON loss_steps(experiment_id);

	CREATE TABLE IF NOT EXISTS inferences (
		id SERIAL PRIMARY KEY,
		experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
		prompt TEXT NOT NULL DEFAULT '',
		output TEXT NOT NULL,
		temperature REAL NOT NULL DEFAULT 0.8,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	);
`;

export async function getDb(): Promise<PGlite> {
	if (db) return db;
	db = new PGlite('idb://autoresearch');
	await db.exec(SCHEMA);
	return db;
}

// -- Experiments --

export type ExperimentRow = {
	id: number;
	name: string;
	source: 'manual' | 'auto';
	code: string;
	reasoning: string;
	val_bpb: number;
	elapsed: number;
	total_steps: number;
	kept: boolean;
	error: string | null;
	loss_curve: { step: number; loss: number }[] | null;
	weights_path: string | null;
	created_at: string;
};

export async function insertExperiment(exp: {
	name?: string;
	source?: 'manual' | 'auto';
	code: string;
	valBpb: number;
	elapsed: number;
	totalSteps: number;
	reasoning: string;
	kept: boolean;
	lossCurve?: { step: number; loss: number }[];
	error?: string;
	createdAt?: string;
}): Promise<number> {
	const pg = await getDb();
	const result = await pg.query<{ id: number }>(
		`INSERT INTO experiments (name, source, code, val_bpb, elapsed, total_steps, reasoning, kept, loss_curve, error, created_at)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, COALESCE($11, NOW()))
		 RETURNING id`,
		[
			exp.name || '',
			exp.source || 'manual',
			exp.code,
			exp.valBpb,
			exp.elapsed,
			exp.totalSteps,
			exp.reasoning,
			exp.kept,
			exp.lossCurve ? JSON.stringify(exp.lossCurve) : null,
			exp.error || null,
			exp.createdAt || null
		]
	);
	return result.rows[0].id;
}

export async function getAllExperiments(): Promise<ExperimentRow[]> {
	const pg = await getDb();
	const result = await pg.query<ExperimentRow>(`SELECT * FROM experiments ORDER BY id`);
	return result.rows;
}

export async function getBestExperiment(): Promise<ExperimentRow | null> {
	const pg = await getDb();
	const result = await pg.query<ExperimentRow>(
		`SELECT * FROM experiments WHERE error IS NULL ORDER BY val_bpb ASC LIMIT 1`
	);
	return result.rows[0] ?? null;
}

export async function updateWeightsPath(id: number, weightsPath: string): Promise<void> {
	const pg = await getDb();
	await pg.query(`UPDATE experiments SET weights_path = $1 WHERE id = $2`, [weightsPath, id]);
}

export async function clearAllData(): Promise<void> {
	const pg = await getDb();
	await clearSavedWeights();
	await pg.exec(`
		DROP TABLE IF EXISTS inferences;
		DROP TABLE IF EXISTS loss_steps;
		DROP TABLE IF EXISTS experiments;
	`);
	await pg.exec(SCHEMA);
}

// -- Loss Steps --

export async function insertLossCurve(experimentId: number, curve: { step: number; loss: number }[]): Promise<void> {
	if (curve.length === 0) return;
	const pg = await getDb();
	const values = curve.map((_, i) => `($1, $${i * 2 + 2}, $${i * 2 + 3})`).join(',');
	const params: (number)[] = [experimentId];
	for (const point of curve) {
		params.push(point.step, point.loss);
	}
	await pg.query(`INSERT INTO loss_steps (experiment_id, step, loss) VALUES ${values}`, params);
}

export async function getAllLossCurves(): Promise<Map<number, { step: number; loss: number }[]>> {
	const pg = await getDb();
	const result = await pg.query<{ experiment_id: number; step: number; loss: number }>(
		`SELECT experiment_id, step, loss FROM loss_steps ORDER BY experiment_id, step`
	);
	const map = new Map<number, { step: number; loss: number }[]>();
	for (const row of result.rows) {
		if (!map.has(row.experiment_id)) map.set(row.experiment_id, []);
		map.get(row.experiment_id)!.push({ step: row.step, loss: row.loss });
	}
	return map;
}

// -- Inferences --

export type InferenceRow = {
	id: number;
	experiment_id: number;
	prompt: string;
	output: string;
	temperature: number;
	created_at: string;
};

export async function insertInference(inf: {
	experimentId: number;
	prompt: string;
	output: string;
	temperature: number;
	createdAt?: string;
}): Promise<number> {
	const pg = await getDb();
	const result = await pg.query<{ id: number }>(
		`INSERT INTO inferences (experiment_id, prompt, output, temperature, created_at)
		 VALUES ($1, $2, $3, $4, COALESCE($5, NOW())) RETURNING id`,
		[inf.experimentId, inf.prompt, inf.output, inf.temperature, inf.createdAt || null]
	);
	return result.rows[0].id;
}

export async function getInferencesForExperiment(experimentId: number): Promise<InferenceRow[]> {
	const pg = await getDb();
	const result = await pg.query<InferenceRow>(
		`SELECT * FROM inferences WHERE experiment_id = $1 ORDER BY created_at DESC`,
		[experimentId]
	);
	return result.rows;
}

// -- Helpers --

import type { ExperimentRecord } from './research/prompt';

export function rowToRecord(row: ExperimentRow): ExperimentRecord {
	return {
		id: row.id,
		name: row.name,
		source: row.source,
		code: row.code,
		valBpb: row.val_bpb,
		elapsed: row.elapsed,
		totalSteps: row.total_steps,
		reasoning: row.reasoning,
		kept: row.kept,
		error: row.error ?? undefined,
		lossCurve: row.loss_curve ?? undefined
	};
}

export async function getAllExperimentRecords(): Promise<ExperimentRecord[]> {
	const rows = await getAllExperiments();
	const lossCurvesMap = await getAllLossCurves();
	return rows.map(row => ({
		...rowToRecord(row),
		lossCurve: lossCurvesMap.get(row.id) ?? row.loss_curve ?? undefined
	}));
}

// -- Export --

function toCsv(rows: Record<string, unknown>[]): string {
	if (rows.length === 0) return '';
	const keys = Object.keys(rows[0]);
	const escape = (v: unknown) => {
		const s = typeof v === 'object' ? JSON.stringify(v) : String(v ?? '');
		return s.includes(',') || s.includes('"') || s.includes('\n')
			? `"${s.replace(/"/g, '""')}"` : s;
	};
	const header = keys.join(',');
	const lines = rows.map(row => keys.map(k => escape(row[k])).join(','));
	return [header, ...lines].join('\n');
}

export async function exportCsvZip(): Promise<Blob> {
	const JSZip = (await import('jszip')).default;
	const pg = await getDb();
	const exps = await pg.query(`SELECT * FROM experiments ORDER BY id`);
	const steps = await pg.query(`SELECT * FROM loss_steps ORDER BY experiment_id, step`);
	const infs = await pg.query(`SELECT * FROM inferences ORDER BY experiment_id, created_at`);
	const zip = new JSZip();
	zip.file('experiments.csv', toCsv(exps.rows as Record<string, unknown>[]));
	zip.file('loss_steps.csv', toCsv(steps.rows as Record<string, unknown>[]));
	zip.file('inferences.csv', toCsv(infs.rows as Record<string, unknown>[]));
	return zip.generateAsync({ type: 'blob' });
}

type CsvRow = Record<string, string>;

type ImportSummary = {
	addedExperiments: number;
	skippedExperiments: number;
	addedLossSteps: number;
	skippedLossSteps: number;
	addedInferences: number;
	skippedInferences: number;
};

type ImportedExperiment = {
	id: number;
	name: string;
	source: 'manual' | 'auto';
	code: string;
	reasoning: string;
	val_bpb: number;
	elapsed: number;
	total_steps: number;
	kept: boolean;
	error: string | null;
	loss_curve: { step: number; loss: number }[] | null;
	created_at: string;
};

type ImportedLossStep = {
	experiment_id: number;
	step: number;
	loss: number;
};

type ImportedInference = {
	experiment_id: number;
	prompt: string;
	output: string;
	temperature: number;
	created_at: string;
};

function chunk<T>(items: T[], size: number): T[][] {
	const chunks: T[][] = [];
	for (let i = 0; i < items.length; i += size) {
		chunks.push(items.slice(i, i + size));
	}
	return chunks;
}

function parseCsv(text: string): CsvRow[] {
	if (!text.trim()) return [];

	const rows: string[][] = [];
	let currentRow: string[] = [];
	let currentField = '';
	let inQuotes = false;

	for (let i = 0; i < text.length; i++) {
		const char = text[i];

		if (inQuotes) {
			if (char === '"') {
				if (text[i + 1] === '"') {
					currentField += '"';
					i++;
				} else {
					inQuotes = false;
				}
			} else {
				currentField += char;
			}
			continue;
		}

		if (char === '"') {
			inQuotes = true;
		} else if (char === ',') {
			currentRow.push(currentField);
			currentField = '';
		} else if (char === '\n') {
			currentRow.push(currentField);
			rows.push(currentRow);
			currentRow = [];
			currentField = '';
		} else if (char !== '\r') {
			currentField += char;
		}
	}

	currentRow.push(currentField);
	if (currentRow.length > 1 || currentRow[0] !== '') {
		rows.push(currentRow);
	}

	if (rows.length === 0) return [];
	const [header, ...dataRows] = rows;
	return dataRows
		.filter((row) => row.some((value) => value !== ''))
		.map((row) => Object.fromEntries(header.map((key, idx) => [key, row[idx] ?? ''])));
}

function parseJsonField<T>(value: string, fallback: T): T {
	if (!value) return fallback;
	try {
		return JSON.parse(value) as T;
	} catch {
		return fallback;
	}
}

function parseNumber(value: string, field: string): number {
	const parsed = Number(value);
	if (Number.isNaN(parsed)) {
		throw new Error(`Invalid number for ${field}: ${value}`);
	}
	return parsed;
}

function parseBoolean(value: string): boolean {
	return value === 'true' || value === 't' || value === '1';
}

function experimentKey(exp: Pick<ExperimentRow, 'name' | 'source' | 'code' | 'reasoning' | 'val_bpb' | 'elapsed' | 'total_steps' | 'kept' | 'error' | 'created_at'>): string {
	return JSON.stringify([
		exp.name,
		exp.source,
		exp.code,
		exp.reasoning,
		exp.val_bpb,
		exp.elapsed,
		exp.total_steps,
		exp.kept,
		exp.error ?? null,
		exp.created_at,
	]);
}

function lossStepKey(experimentId: number, step: number, loss: number): string {
	return `${experimentId}:${step}:${loss}`;
}

function inferenceKey(inf: Pick<InferenceRow, 'experiment_id' | 'prompt' | 'output' | 'temperature' | 'created_at'>): string {
	return JSON.stringify([inf.experiment_id, inf.prompt, inf.output, inf.temperature, inf.created_at]);
}

function parseImportedExperiments(rows: CsvRow[]): ImportedExperiment[] {
	return rows.map((row) => ({
		id: parseNumber(row.id, 'experiments.id'),
		name: row.name ?? '',
		source: row.source === 'auto' ? 'auto' : 'manual',
		code: row.code ?? '',
		reasoning: row.reasoning ?? '',
		val_bpb: parseNumber(row.val_bpb, 'experiments.val_bpb'),
		elapsed: parseNumber(row.elapsed, 'experiments.elapsed'),
		total_steps: parseNumber(row.total_steps, 'experiments.total_steps'),
		kept: parseBoolean(row.kept ?? ''),
		error: row.error ? row.error : null,
		loss_curve: parseJsonField(row.loss_curve ?? '', null),
		created_at: row.created_at || new Date().toISOString(),
	}));
}

function parseImportedLossSteps(rows: CsvRow[]): ImportedLossStep[] {
	return rows.map((row) => ({
		experiment_id: parseNumber(row.experiment_id, 'loss_steps.experiment_id'),
		step: parseNumber(row.step, 'loss_steps.step'),
		loss: parseNumber(row.loss, 'loss_steps.loss'),
	}));
}

function parseImportedInferences(rows: CsvRow[]): ImportedInference[] {
	return rows.map((row) => ({
		experiment_id: parseNumber(row.experiment_id, 'inferences.experiment_id'),
		prompt: row.prompt ?? '',
		output: row.output ?? '',
		temperature: parseNumber(row.temperature, 'inferences.temperature'),
		created_at: row.created_at || new Date().toISOString(),
	}));
}

async function insertExperimentsBatch(pg: PGlite, experiments: ImportedExperiment[]): Promise<number[]> {
	if (experiments.length === 0) return [];

	const values = experiments
		.map((_, idx) => {
			const base = idx * 11;
			return `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4}, $${base + 5}, $${base + 6}, $${base + 7}, $${base + 8}, $${base + 9}, $${base + 10}, COALESCE($${base + 11}, NOW()))`;
		})
		.join(', ');

	const params: unknown[] = [];
	for (const exp of experiments) {
		params.push(
			exp.name,
			exp.source,
			exp.code,
			exp.val_bpb,
			exp.elapsed,
			exp.total_steps,
			exp.reasoning,
			exp.kept,
			exp.loss_curve ? JSON.stringify(exp.loss_curve) : null,
			exp.error,
			exp.created_at
		);
	}

	const result = await pg.query<{ id: number }>(
		`INSERT INTO experiments (name, source, code, val_bpb, elapsed, total_steps, reasoning, kept, loss_curve, error, created_at)
		 VALUES ${values}
		 RETURNING id`,
		params
	);

	return result.rows.map((row) => row.id);
}

async function insertLossStepsBatch(pg: PGlite, rows: ImportedLossStep[], idMap: Map<number, number>): Promise<number> {
	let inserted = 0;

	for (const batch of chunk(rows, 500)) {
		const mapped = batch
			.map((row) => ({
				experimentId: idMap.get(row.experiment_id),
				step: row.step,
				loss: row.loss,
			}))
			.filter((row): row is { experimentId: number; step: number; loss: number } => row.experimentId != null);

		if (mapped.length === 0) continue;

		const values = mapped
			.map((_, idx) => {
				const base = idx * 3;
				return `($${base + 1}, $${base + 2}, $${base + 3})`;
			})
			.join(', ');

		const params: number[] = [];
		for (const row of mapped) {
			params.push(row.experimentId, row.step, row.loss);
		}

		await pg.query(`INSERT INTO loss_steps (experiment_id, step, loss) VALUES ${values}`, params);
		inserted += mapped.length;
	}

	return inserted;
}

async function insertInferencesBatch(pg: PGlite, rows: ImportedInference[], idMap: Map<number, number>): Promise<number> {
	let inserted = 0;

	for (const batch of chunk(rows, 200)) {
		const mapped = batch
			.map((row) => ({
				experimentId: idMap.get(row.experiment_id),
				prompt: row.prompt,
				output: row.output,
				temperature: row.temperature,
				createdAt: row.created_at,
			}))
			.filter((row): row is { experimentId: number; prompt: string; output: string; temperature: number; createdAt: string } => row.experimentId != null);

		if (mapped.length === 0) continue;

		const values = mapped
			.map((_, idx) => {
				const base = idx * 5;
				return `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4}, COALESCE($${base + 5}, NOW()))`;
			})
			.join(', ');

		const params: unknown[] = [];
		for (const row of mapped) {
			params.push(row.experimentId, row.prompt, row.output, row.temperature, row.createdAt);
		}

		await pg.query(
			`INSERT INTO inferences (experiment_id, prompt, output, temperature, created_at)
			 VALUES ${values}`,
			params
		);
		inserted += mapped.length;
	}

	return inserted;
}

export async function importCsvZip(file: Blob): Promise<ImportSummary> {
	const JSZip = (await import('jszip')).default;
	const zip = await JSZip.loadAsync(await file.arrayBuffer());

	const experimentsFile = zip.file('experiments.csv');
	if (!experimentsFile) {
		throw new Error('Import ZIP is missing experiments.csv');
	}

	const [experimentsCsv, lossStepsCsv, inferencesCsv] = await Promise.all([
		experimentsFile.async('string'),
		zip.file('loss_steps.csv')?.async('string') ?? Promise.resolve(''),
		zip.file('inferences.csv')?.async('string') ?? Promise.resolve(''),
	]);

	const importedExperiments = parseImportedExperiments(parseCsv(experimentsCsv)).sort((a, b) => a.id - b.id);
	const importedLossSteps = parseImportedLossSteps(parseCsv(lossStepsCsv));
	const importedInferences = parseImportedInferences(parseCsv(inferencesCsv));

	const summary: ImportSummary = {
		addedExperiments: 0,
		skippedExperiments: 0,
		addedLossSteps: 0,
		skippedLossSteps: 0,
		addedInferences: 0,
		skippedInferences: 0,
	};

	const pg = await getDb();
	const existingExperiments = await getAllExperiments();
	const existingExperimentKeys = new Map(existingExperiments.map((row) => [experimentKey(row), row.id]));

	const existingLossStepRows = await pg.query<{ experiment_id: number; step: number; loss: number }>(
		`SELECT experiment_id, step, loss FROM loss_steps`
	);
	const existingLossStepKeys = new Set(
		existingLossStepRows.rows.map((row) => lossStepKey(row.experiment_id, row.step, row.loss))
	);

	const existingInferenceRows = await pg.query<InferenceRow>(
		`SELECT experiment_id, prompt, output, temperature, created_at FROM inferences`
	);
	const existingInferenceKeys = new Set(existingInferenceRows.rows.map(inferenceKey));

	const idMap = new Map<number, number>();

	await pg.exec('BEGIN');
	try {
		const experimentsToInsert: ImportedExperiment[] = [];

		for (const exp of importedExperiments) {
			const key = experimentKey(exp);
			const existingId = existingExperimentKeys.get(key);
			if (existingId != null) {
				idMap.set(exp.id, existingId);
				summary.skippedExperiments++;
				continue;
			}

			experimentsToInsert.push(exp);
		}

		for (const batch of chunk(experimentsToInsert, 50)) {
			const insertedIds = await insertExperimentsBatch(pg, batch);
			for (let i = 0; i < batch.length; i++) {
				const exp = batch[i];
				const insertedId = insertedIds[i];
				idMap.set(exp.id, insertedId);
				existingExperimentKeys.set(experimentKey(exp), insertedId);
				summary.addedExperiments++;
			}
		}

		const lossStepsToInsert: ImportedLossStep[] = [];
		for (const step of importedLossSteps) {
			const mappedExperimentId = idMap.get(step.experiment_id);
			if (mappedExperimentId == null) continue;

			const key = lossStepKey(mappedExperimentId, step.step, step.loss);
			if (existingLossStepKeys.has(key)) {
				summary.skippedLossSteps++;
				continue;
			}

			existingLossStepKeys.add(key);
			lossStepsToInsert.push(step);
		}
		summary.addedLossSteps += await insertLossStepsBatch(pg, lossStepsToInsert, idMap);

		const inferencesToInsert: ImportedInference[] = [];
		for (const inf of importedInferences) {
			const mappedExperimentId = idMap.get(inf.experiment_id);
			if (mappedExperimentId == null) continue;

			const key = inferenceKey({
				experiment_id: mappedExperimentId,
				prompt: inf.prompt,
				output: inf.output,
				temperature: inf.temperature,
				created_at: inf.created_at,
			});
			if (existingInferenceKeys.has(key)) {
				summary.skippedInferences++;
				continue;
			}

			existingInferenceKeys.add(key);
			inferencesToInsert.push(inf);
		}
		summary.addedInferences += await insertInferencesBatch(pg, inferencesToInsert, idMap);

		await pg.exec('COMMIT');
		return summary;
	} catch (error) {
		await pg.exec('ROLLBACK');
		throw error;
	}
}
