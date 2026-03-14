import { PGlite } from '@electric-sql/pglite';

let db: PGlite | null = null;

export async function getDb(): Promise<PGlite> {
	if (db) return db;
	db = new PGlite('idb://autoresearch');
	await db.exec(`
		CREATE TABLE IF NOT EXISTS experiments (
			id SERIAL PRIMARY KEY,
			name TEXT NOT NULL DEFAULT '',
			source TEXT NOT NULL DEFAULT 'manual',
			config JSONB NOT NULL,
			val_bpb REAL NOT NULL,
			elapsed REAL NOT NULL,
			total_steps INTEGER NOT NULL,
			reasoning TEXT NOT NULL DEFAULT '',
			kept BOOLEAN NOT NULL DEFAULT false,
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
	`);
	return db;
}

// -- Experiments --

export type ExperimentRow = {
	id: number;
	name: string;
	source: 'manual' | 'auto';
	config: Record<string, unknown>;
	val_bpb: number;
	elapsed: number;
	total_steps: number;
	reasoning: string;
	kept: boolean;
	loss_curve: { step: number; loss: number }[] | null;
	weights_path: string | null;
	created_at: string;
};

export async function insertExperiment(exp: {
	name?: string;
	source?: 'manual' | 'auto';
	config: Record<string, unknown>;
	valBpb: number;
	elapsed: number;
	totalSteps: number;
	reasoning: string;
	kept: boolean;
	lossCurve?: { step: number; loss: number }[];
}): Promise<number> {
	const pg = await getDb();
	const result = await pg.query<{ id: number }>(
		`INSERT INTO experiments (name, source, config, val_bpb, elapsed, total_steps, reasoning, kept, loss_curve)
		 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		 RETURNING id`,
		[
			exp.name || '',
			exp.source || 'manual',
			JSON.stringify(exp.config),
			exp.valBpb,
			exp.elapsed,
			exp.totalSteps,
			exp.reasoning,
			exp.kept,
			exp.lossCurve ? JSON.stringify(exp.lossCurve) : null
		]
	);
	return result.rows[0].id;
}

export async function getAllExperiments(): Promise<ExperimentRow[]> {
	const pg = await getDb();
	const result = await pg.query<ExperimentRow>(
		`SELECT * FROM experiments ORDER BY id`
	);
	return result.rows;
}

export async function getBestExperiment(): Promise<ExperimentRow | null> {
	const pg = await getDb();
	const result = await pg.query<ExperimentRow>(
		`SELECT * FROM experiments ORDER BY val_bpb ASC LIMIT 1`
	);
	return result.rows[0] ?? null;
}

export async function updateWeightsPath(id: number, weightsPath: string): Promise<void> {
	const pg = await getDb();
	await pg.query(`UPDATE experiments SET weights_path = $1 WHERE id = $2`, [weightsPath, id]);
}

export async function clearAllData(): Promise<void> {
	const pg = await getDb();
	await pg.exec(`DELETE FROM inferences; DELETE FROM loss_steps; DELETE FROM experiments;`);
}

// -- Loss Steps --

export async function insertLossCurve(experimentId: number, curve: { step: number; loss: number }[]): Promise<void> {
	if (curve.length === 0) return;
	const pg = await getDb();
	// Batch insert
	const values = curve.map((_, i) => `($1, $${i * 2 + 2}, $${i * 2 + 3})`).join(',');
	const params: (number)[] = [experimentId];
	for (const point of curve) {
		params.push(point.step, point.loss);
	}
	await pg.query(`INSERT INTO loss_steps (experiment_id, step, loss) VALUES ${values}`, params);
}

export async function getLossCurve(experimentId: number): Promise<{ step: number; loss: number }[]> {
	const pg = await getDb();
	const result = await pg.query<{ step: number; loss: number }>(
		`SELECT step, loss FROM loss_steps WHERE experiment_id = $1 ORDER BY step`,
		[experimentId]
	);
	return result.rows;
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
		 VALUES ($1, $2, $3, $4)
		 RETURNING id`,
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

export async function exportExperimentsJson(): Promise<string> {
	const pg = await getDb();
	const exps = await pg.query<ExperimentRow>(
		`SELECT id, config, val_bpb, elapsed, total_steps, reasoning, kept, created_at
		 FROM experiments ORDER BY id`
	);
	const infs = await pg.query<InferenceRow>(
		`SELECT * FROM inferences ORDER BY experiment_id, created_at`
	);
	return JSON.stringify({ experiments: exps.rows, inferences: infs.rows }, null, 2);
}
