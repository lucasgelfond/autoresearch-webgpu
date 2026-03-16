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
}): Promise<number> {
	const pg = await getDb();
	const result = await pg.query<{ id: number }>(
		`INSERT INTO experiments (name, source, code, val_bpb, elapsed, total_steps, reasoning, kept, loss_curve, error)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
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
			exp.error || null
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
}): Promise<number> {
	const pg = await getDb();
	const result = await pg.query<{ id: number }>(
		`INSERT INTO inferences (experiment_id, prompt, output, temperature)
		 VALUES ($1, $2, $3, $4) RETURNING id`,
		[inf.experimentId, inf.prompt, inf.output, inf.temperature]
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
