<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { EditorView } from '@codemirror/view';
	import { EditorState, Compartment } from '@codemirror/state';
	import { javascript } from '@codemirror/lang-javascript';
	import { basicSetup } from 'codemirror';
	import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
	import { tags } from '@lezer/highlight';

	let { value = $bindable(), disabled = false }: { value: string; disabled?: boolean } = $props();

	let container: HTMLDivElement;
	let view: EditorView;
	let updating = false;
	const readOnlyComp = new Compartment();

	const appTheme = EditorView.theme({
		'&': { fontSize: '10.5px', height: '100%', background: 'transparent' },
		'.cm-scroller': { overflow: 'auto', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' },
		'.cm-content': { padding: '8px 0', caretColor: '#93c5fd' },
		'.cm-gutters': { fontSize: '9px', minWidth: '28px', background: 'transparent', color: '#4b5563', border: 'none' },
		'.cm-activeLineGutter': { background: 'rgba(59, 130, 246, 0.08)' },
		'.cm-activeLine': { background: 'rgba(59, 130, 246, 0.06)' },
		'.cm-selectionBackground': { background: 'rgba(59, 130, 246, 0.2) !important' },
		'&.cm-focused .cm-selectionBackground': { background: 'rgba(59, 130, 246, 0.3) !important' },
		'.cm-cursor': { borderLeftColor: '#93c5fd' },
		'.cm-matchingBracket': { background: 'rgba(59, 130, 246, 0.25)', outline: 'none' },
		'.cm-foldGutter': { color: '#4b5563' },
		'.cm-tooltip': { background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' },
		'.cm-tooltip-autocomplete': { background: '#1e293b' },
	}, { dark: true });

	const highlighting = HighlightStyle.define([
		{ tag: tags.keyword, color: '#c084fc' },
		{ tag: tags.controlKeyword, color: '#c084fc' },
		{ tag: tags.definitionKeyword, color: '#c084fc' },
		{ tag: tags.operatorKeyword, color: '#c084fc' },
		{ tag: tags.variableName, color: '#e2e8f0' },
		{ tag: tags.definition(tags.variableName), color: '#93c5fd' },
		{ tag: tags.function(tags.variableName), color: '#67e8f9' },
		{ tag: tags.propertyName, color: '#6ee7b7' },
		{ tag: tags.definition(tags.propertyName), color: '#6ee7b7' },
		{ tag: tags.number, color: '#fbbf24' },
		{ tag: tags.string, color: '#86efac' },
		{ tag: tags.bool, color: '#fbbf24' },
		{ tag: tags.null, color: '#fbbf24' },
		{ tag: tags.operator, color: '#94a3b8' },
		{ tag: tags.punctuation, color: '#64748b' },
		{ tag: tags.comment, color: '#475569', fontStyle: 'italic' },
		{ tag: tags.lineComment, color: '#475569', fontStyle: 'italic' },
		{ tag: tags.blockComment, color: '#475569', fontStyle: 'italic' },
		{ tag: tags.typeName, color: '#f0abfc' },
		{ tag: tags.atom, color: '#fbbf24' },
	]);

	onMount(() => {
		const state = EditorState.create({
			doc: value,
			extensions: [
				basicSetup,
				javascript(),
				appTheme,
				syntaxHighlighting(highlighting),
				readOnlyComp.of(EditorState.readOnly.of(disabled)),
				EditorView.updateListener.of((update) => {
					if (update.docChanged && !updating) {
						value = update.state.doc.toString();
					}
				}),
			],
		});
		view = new EditorView({ state, parent: container });
	});

	onDestroy(() => view?.destroy());

	// Sync value → editor
	$effect(() => {
		if (view && value !== view.state.doc.toString()) {
			updating = true;
			view.dispatch({
				changes: { from: 0, to: view.state.doc.length, insert: value },
			});
			updating = false;
			const scroller = view.scrollDOM;
			scroller.scrollTop = scroller.scrollHeight;
			scroller.scrollLeft = 0;
		}
	});

	// Sync disabled → readOnly
	$effect(() => {
		if (view) {
			view.dispatch({
				effects: readOnlyComp.reconfigure(EditorState.readOnly.of(disabled)),
			});
		}
	});
</script>

<div bind:this={container} class="h-full overflow-hidden rounded border border-gray-800 bg-gray-950"></div>
