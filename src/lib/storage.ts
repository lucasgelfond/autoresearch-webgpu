import type { ExperimentRecord } from './research/prompt';
import type { ExperimentConfig } from './model/config';

const DB_NAME = 'autoresearch';
const DB_VERSION = 1;
const STORE_NAME = 'experiments';
const META_STORE = 'meta';

function openDb(): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const req = indexedDB.open(DB_NAME, DB_VERSION);
		req.onerror = () => reject(req.error);
		req.onsuccess = () => resolve(req.result);
		req.onupgradeneeded = () => {
			const db = req.result;
			if (!db.objectStoreNames.contains(STORE_NAME)) {
				db.createObjectStore(STORE_NAME, { keyPath: 'id' });
			}
			if (!db.objectStoreNames.contains(META_STORE)) {
				db.createObjectStore(META_STORE);
			}
		};
	});
}

export async function saveExperiment(record: ExperimentRecord): Promise<void> {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(STORE_NAME, 'readwrite');
		tx.objectStore(STORE_NAME).put(record);
		tx.oncomplete = () => resolve();
		tx.onerror = () => reject(tx.error);
	});
}

export async function loadExperiments(): Promise<ExperimentRecord[]> {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(STORE_NAME, 'readonly');
		const req = tx.objectStore(STORE_NAME).getAll();
		req.onsuccess = () => resolve(req.result);
		req.onerror = () => reject(req.error);
	});
}

export async function saveBestConfig(config: ExperimentConfig, bpb: number): Promise<void> {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(META_STORE, 'readwrite');
		const store = tx.objectStore(META_STORE);
		store.put(config, 'bestConfig');
		store.put(bpb, 'bestBpb');
		tx.oncomplete = () => resolve();
		tx.onerror = () => reject(tx.error);
	});
}

export async function loadBestConfig(): Promise<{ config: ExperimentConfig; bpb: number } | null> {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction(META_STORE, 'readonly');
		const store = tx.objectStore(META_STORE);
		const configReq = store.get('bestConfig');
		const bpbReq = store.get('bestBpb');
		tx.oncomplete = () => {
			if (configReq.result && bpbReq.result) {
				resolve({ config: configReq.result, bpb: bpbReq.result });
			} else {
				resolve(null);
			}
		};
		tx.onerror = () => reject(tx.error);
	});
}

export async function clearAll(): Promise<void> {
	const db = await openDb();
	return new Promise((resolve, reject) => {
		const tx = db.transaction([STORE_NAME, META_STORE], 'readwrite');
		tx.objectStore(STORE_NAME).clear();
		tx.objectStore(META_STORE).clear();
		tx.oncomplete = () => resolve();
		tx.onerror = () => reject(tx.error);
	});
}

export function exportAsJson(experiments: ExperimentRecord[]): string {
	const slim = experiments.map(({ lossCurve, ...rest }) => rest);
	return JSON.stringify(slim, null, 2);
}

export function exportAsTsv(experiments: ExperimentRecord[]): string {
	const header = 'id\tval_bpb\tsteps\telapsed_s\tkept\treasoning';
	const rows = experiments.map(
		(e) =>
			`${e.id}\t${e.valBpb.toFixed(4)}\t${e.totalSteps}\t${(e.elapsed / 1000).toFixed(1)}\t${e.kept}\t${e.reasoning}`
	);
	return [header, ...rows].join('\n');
}
