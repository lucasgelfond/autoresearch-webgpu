<script lang="ts">
	import {
		createResearchEndpointProfile,
		defaultBaseUrl,
		isConfiguredProfile,
		type ResearchEndpointProfile,
		type ResearchProvider,
	} from '$lib/research/providers';

	let {
		profiles = $bindable(),
		selectedId = $bindable(),
		disabled = false,
	}: {
		profiles: ResearchEndpointProfile[];
		selectedId: string | null;
		disabled?: boolean;
	} = $props();

	let selectedProfile = $derived(
		selectedId ? profiles.find((profile) => profile.id === selectedId) ?? null : null
	);
	let selectedConfigured = $derived(isConfiguredProfile(selectedProfile));

	function setSelectedProfile(id: string) {
		selectedId = id;
	}

	function updateSelectedProfile(
		patch: Partial<ResearchEndpointProfile>,
		options: { syncDefaultUrl?: boolean } = {}
	) {
		if (!selectedProfile) return;

		profiles = profiles.map((profile) => {
			if (profile.id !== selectedProfile!.id) return profile;

			const next = { ...profile, ...patch };
			if (options.syncDefaultUrl && patch.provider) {
				const previousDefault = defaultBaseUrl(profile.provider);
				if (!profile.baseUrl || profile.baseUrl === previousDefault) {
					next.baseUrl = defaultBaseUrl(patch.provider);
				}
			}

			return next;
		});
	}

	function addProfile(provider: ResearchProvider) {
		const next = createResearchEndpointProfile(provider);
		profiles = [...profiles, next];
		selectedId = next.id;
	}

	function removeSelectedProfile() {
		if (!selectedProfile || profiles.length === 1) return;

		const idx = profiles.findIndex((profile) => profile.id === selectedProfile.id);
		const remaining = profiles.filter((profile) => profile.id !== selectedProfile.id);
		profiles = remaining;
		selectedId = remaining[Math.max(0, idx - 1)]?.id ?? remaining[0]?.id ?? null;
	}

	function profileBadge(profile: ResearchEndpointProfile): string {
		return profile.provider === 'anthropic' ? 'anthropic' : 'openai';
	}
</script>

<div class="rounded border border-gray-800 bg-gray-950/60 p-3 space-y-3">
	<div class="flex items-start justify-between gap-3">
		<div>
			<h2 class="text-xs font-mono text-gray-300">research backends</h2>
			<p class="text-[10px] font-mono text-gray-500 mt-1">
				keys are stored locally in this browser only.
			</p>
		</div>
		<div class="flex gap-1 shrink-0">
			<button
				type="button"
				class="rounded border border-gray-700 px-2 py-1 text-[10px] font-mono text-gray-300 hover:border-blue-500 hover:text-white disabled:opacity-40"
				onclick={() => addProfile('anthropic')}
				disabled={disabled}
			>+ anthropic</button>
			<button
				type="button"
				class="rounded border border-gray-700 px-2 py-1 text-[10px] font-mono text-gray-300 hover:border-blue-500 hover:text-white disabled:opacity-40"
				onclick={() => addProfile('openai')}
				disabled={disabled}
			>+ openai</button>
		</div>
	</div>

	<div class="flex flex-wrap gap-1.5">
		{#each profiles as profile}
			<button
				type="button"
				class="rounded border px-2 py-1 text-[10px] font-mono transition-colors
					{selectedId === profile.id ? 'border-blue-500 bg-blue-950/50 text-blue-200' : 'border-gray-800 text-gray-400 hover:border-gray-600 hover:text-gray-200'}"
				onclick={() => setSelectedProfile(profile.id)}
				disabled={disabled}
			>
				{profile.name || 'unnamed'}
				<span class="ml-1 text-[9px] uppercase tracking-wide opacity-70">{profileBadge(profile)}</span>
			</button>
		{/each}
	</div>

	{#if selectedProfile}
		<div class="grid grid-cols-1 md:grid-cols-2 gap-2">
			<label class="space-y-1">
				<span class="block text-[10px] font-mono text-gray-500">name</span>
				<input
					type="text"
					value={selectedProfile.name}
					oninput={(e) => updateSelectedProfile({ name: (e.currentTarget as HTMLInputElement).value })}
					disabled={disabled}
					class="w-full rounded border border-gray-800 bg-black/40 px-2 py-1.5 text-xs font-mono text-gray-200 disabled:opacity-40"
				/>
			</label>
			<label class="space-y-1">
				<span class="block text-[10px] font-mono text-gray-500">provider</span>
				<select
					value={selectedProfile.provider}
					onchange={(e) => updateSelectedProfile(
						{ provider: (e.currentTarget as HTMLSelectElement).value as ResearchProvider },
						{ syncDefaultUrl: true }
					)}
					disabled={disabled}
					class="w-full rounded border border-gray-800 bg-black/40 px-2 py-1.5 text-xs font-mono text-gray-200 disabled:opacity-40"
				>
					<option value="anthropic">anthropic</option>
					<option value="openai">openai compatible</option>
				</select>
			</label>
			<label class="space-y-1 md:col-span-2">
				<span class="block text-[10px] font-mono text-gray-500">endpoint url</span>
				<input
					type="url"
					value={selectedProfile.baseUrl}
					oninput={(e) => updateSelectedProfile({ baseUrl: (e.currentTarget as HTMLInputElement).value })}
					disabled={disabled}
					class="w-full rounded border border-gray-800 bg-black/40 px-2 py-1.5 text-xs font-mono text-gray-200 disabled:opacity-40"
				/>
			</label>
			<label class="space-y-1">
				<span class="block text-[10px] font-mono text-gray-500">model</span>
				<input
					type="text"
					value={selectedProfile.model}
					oninput={(e) => updateSelectedProfile({ model: (e.currentTarget as HTMLInputElement).value })}
					disabled={disabled}
					class="w-full rounded border border-gray-800 bg-black/40 px-2 py-1.5 text-xs font-mono text-gray-200 disabled:opacity-40"
				/>
			</label>
			<label class="space-y-1">
				<span class="block text-[10px] font-mono text-gray-500">api key</span>
				<input
					type="password"
					value={selectedProfile.apiKey}
					oninput={(e) => updateSelectedProfile({ apiKey: (e.currentTarget as HTMLInputElement).value })}
					disabled={disabled}
					class="w-full rounded border border-gray-800 bg-black/40 px-2 py-1.5 text-xs font-mono text-gray-200 disabled:opacity-40"
				/>
			</label>
		</div>

		<div class="flex items-center justify-between gap-3">
			<p class="text-[10px] font-mono {selectedConfigured ? 'text-emerald-400' : 'text-amber-400'}">
				{selectedConfigured ? 'selected profile is ready for research runs' : 'fill endpoint, model, and api key to use this profile'}
			</p>
			<button
				type="button"
				class="rounded border border-red-900 px-2 py-1 text-[10px] font-mono text-red-300 hover:border-red-500 hover:text-red-200 disabled:opacity-30"
				onclick={removeSelectedProfile}
				disabled={disabled || profiles.length === 1}
			>delete profile</button>
		</div>
	{/if}
</div>
