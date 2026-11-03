export type GenerateOptions = {
	systemPrompt: string;
	userPrompt: string;
	onToken?: (text: string) => void;
	signal?: AbortSignal;
	temperature?: number;
};

export interface ResearchProvider {
	generate(options: GenerateOptions): Promise<string>;
}
