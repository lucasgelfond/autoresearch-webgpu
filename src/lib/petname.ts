const adjectives = [
	'adorable', 'brave', 'calm', 'dazzling', 'eager', 'fancy', 'gentle', 'happy',
	'icy', 'jolly', 'keen', 'lively', 'mellow', 'noble', 'odd', 'proud',
	'quiet', 'rapid', 'silly', 'tidy', 'unique', 'vivid', 'witty', 'zany',
	'bold', 'crisp', 'dusty', 'fiery', 'glossy', 'hasty', 'itchy', 'jumpy',
	'kind', 'lucky', 'misty', 'neat', 'orange', 'plush', 'quick', 'rosy',
	'sharp', 'tough', 'ultra', 'vast', 'warm', 'young', 'zippy', 'bright',
	'cozy', 'dizzy', 'epic', 'fresh', 'grand', 'husky', 'ivory', 'jade',
];

const animals = [
	'alpaca', 'badger', 'cat', 'dog', 'eagle', 'fox', 'goose', 'hawk',
	'ibis', 'jaguar', 'koala', 'lion', 'moose', 'newt', 'otter', 'panda',
	'quail', 'robin', 'snake', 'tiger', 'urchin', 'viper', 'wolf', 'yak',
	'bear', 'crane', 'dove', 'elk', 'frog', 'gecko', 'heron', 'iguana',
	'jay', 'kiwi', 'lynx', 'mink', 'owl', 'pike', 'raven', 'seal',
	'toad', 'whale', 'zebra', 'bison', 'crab', 'deer', 'emu', 'finch',
];

function pick(arr: string[]): string {
	return arr[Math.floor(Math.random() * arr.length)];
}

export function petname(): string {
	return `${pick(adjectives)}-${pick(animals)}`;
}
