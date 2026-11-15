/**
 * French copy for every user-facing string in the app.
 *
 * Conventions:
 * - plain ASCII apostrophes (') and quotes, so the strings are safe in code and
 *   in HTML attributes;
 * - U+00A0 (no-break space) before French double punctuation ( ? ! : ; ) and
 *   inside « » quotes, as French typography requires;
 * - identifiers, file names (train.ts), metric names (val_bpb, bpb, loss curves),
 *   API/model names (WebGPU, jax-js, Claude, pglite) and log levels stay as-is.
 */

export const fr = {
	meta: {
		title: 'autoresearch, dans le navigateur\u00a0!',
		description:
			"Entraînez de petits modèles de langage dans votre navigateur grâce à WebGPU. Claude écrit le code d'entraînement, lance des expériences et itère — un modèle qui en entraîne un autre\u00a0!",
		ogDescription:
			"Entraînez de petits modèles de langage dans votre navigateur grâce à WebGPU. Claude écrit le code d'entraînement, lance des expériences et itère.",
		twitterDescription:
			"Entraînez de petits modèles de langage dans votre navigateur grâce à WebGPU. Claude écrit le code d'entraînement et itère."
	},

	header: {
		title: 'autoresearch, dans le navigateur\u00a0!',
		tagline:
			"Entraînez un petit modèle de langage (entièrement sur l'appareil, avec WebGPU\u00a0!) en générant du code d'entraînement, en lançant des expériences et en itérant au fil des progrès.",
		credits: {
			beforeAutoresearch: "Inspiré de l'",
			afterAutoresearch: " d'Andrej Karpathy et construit sur ",
			afterJaxJs: " d'",
			afterAuthor: '. Créé par ',
			beforeSource: '. Code source ',
			afterSource: '.',
			autoresearch: 'autoresearch',
			jaxJs: 'jax-js',
			author: 'Lucas Gelfond',
			source: 'ici'
		}
	},

	mobileWarning: {
		label: 'ATTENTION\u00a0:',
		text: "les appareils mobiles plantent souvent lors d'activités web gourmandes en mémoire. Meilleurs résultats sur un ordinateur\u00a0!"
	},

	initError: "une erreur est survenue lors de l'initialisation. consultez la console pour plus de détails.",

	firstRun: {
		start: 'lancer la recherche',
		hint: "Claude va écrire du code d'entraînement, l'exécuter dans votre navigateur et itérer pour trouver de meilleurs modèles."
	},

	mode: {
		manual: 'manuel',
		research: 'auto'
	},

	train: {
		heading: 'train.ts',
		reset: 'réinitialiser',
		namePlaceholder: "nom de l'expérience…",
		stopTraining: "arrêter l'entraînement",
		run: 'lancer train.ts',
		startResearch: 'lancer la recherche',
		stop: 'arrêter'
	},

	experiment: {
		sourceAuto: 'auto',
		sourceManual: 'manuel',
		error: 'erreur'
	},

	run: {
		thinking: 'réflexion',
		running: 'en cours',
		writingCode: 'Claude écrit le code…',
		training: 'entraînement…'
	},

	chart: {
		heading: 'perte',
		waiting: 'en attente de données…',
		xAxis: 'étape'
	},

	inference: {
		heading: 'inférence',
		promptPlaceholder: 'prompt…',
		temperature: 'température',
		go: 'générer',
		generating: 'génération en cours',
		newer: 'inférence plus récente',
		older: 'inférence plus ancienne'
	},

	list: {
		leaderboard: 'classement',
		history: 'historique',
		empty: "aucune expérience pour l'instant",
		inProgress: 'en cours',
		sourceAuto: 'auto (Claude)',
		sourceManual: 'manuel',
		export: 'exporter',
		clear: 'effacer'
	},

	clear: {
		title: 'effacer toutes les données\u00a0?',
		body: "cela supprimera toutes les expériences, les courbes de perte, les inférences et les poids enregistrés. cette action est irréversible.",
		cancel: 'annuler',
		confirm: 'tout effacer'
	},

	status: {
		initializing: 'initialisation…',
		loadingData: 'chargement des données…',
		ready: 'prêt',
		starting: 'démarrage…',
		training: 'entraînement…',
		loadingModel: 'chargement du modèle…',
		step: (step: number, loss: string, seconds: string) =>
			`étape ${step} | perte ${loss} | ${seconds}s`,
		done: (bpb: string, steps: number) => `terminé — val_bpb\u00a0: ${bpb} | ${steps} étapes`,
		experiment: (reasoning: string) => `expérience\u00a0: ${reasoning}`,
		experimentDone: (id: number, kept: boolean, bpb: string) =>
			`#${id} ${kept ? 'retenue' : 'écartée'} — bpb ${bpb}`,
		thinking: (reasoning: string) => `réflexion\u00a0: ${reasoning}`,
		error: (message: string) => `erreur\u00a0: ${message}`
	},

	errors: {
		webgpuUnsupported: "WebGPU n'est pas pris en charge par ce navigateur.",
		noAdapter: 'Aucun adaptateur WebGPU détecté.',
		jaxInit: (msg: string) => `Échec de l'initialisation de jax-js\u00a0: ${msg}`,
		timeout: (seconds: number) => `L'entraînement a dépassé le délai de ${seconds}\u00a0s`,
		noNextCode: "Impossible d'obtenir le prochain code de Claude.",
		emptyResponse: 'Réponse vide de Claude',
		parseFailed: "Impossible d'analyser la réponse de Claude",
		api: (status: number, detail: string) => `API ${status}\u00a0: ${detail}`,
		fetchFailed: (err: unknown) => `Échec de la requête\u00a0: ${err}`,
		server: {
			forbidden: 'Accès refusé',
			apiKeyMissing: 'Clé API non configurée',
			rateLimited: 'Trop de requêtes. Maximum 1 requête toutes les 10 secondes.',
			cannotParse: "Impossible d'analyser la réponse"
		}
	},

	reasoning: {
		manualRun: 'Exécution manuelle',
		baseline: "Exécution de référence avec l'architecture par défaut."
	}
};

export type Messages = typeof fr;
