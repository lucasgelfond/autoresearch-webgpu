<script lang="ts">
	import { t } from '$lib/i18n';

	type Point = { step: number; loss: number };
	type Series = { data: Point[]; color: string; label?: string; highlight?: boolean };

	let { data, pastRuns = [] }: { data: Point[]; pastRuns?: Series[] } = $props();

	let canvas: HTMLCanvasElement;

	const COLORS = ['#6b7280', '#4b5563', '#374151', '#9ca3af', '#6b7280'];

	/** Pick nice round tick values for an axis range. */
	function niceSteps(min: number, max: number, maxTicks: number): number[] {
		const range = max - min;
		if (range <= 0) return [min];
		const rough = range / maxTicks;
		const mag = Math.pow(10, Math.floor(Math.log10(rough)));
		const residual = rough / mag;
		const nice = residual <= 1.5 ? 1 : residual <= 3 ? 2 : residual <= 7 ? 5 : 10;
		const step = nice * mag;
		const start = Math.ceil(min / step) * step;
		const ticks: number[] = [];
		for (let v = start; v <= max + step * 0.01; v += step) {
			ticks.push(v);
		}
		return ticks;
	}

	$effect(() => {
		if (!canvas) return;

		const allSeries: Series[] = [
			...pastRuns.map((r, i) => ({ ...r, color: r.color || COLORS[i % COLORS.length] })),
			...(data.length >= 2 ? [{ data, color: '#3b82f6', label: 'current' }] : [])
		];

		const ctx = canvas.getContext('2d')!;
		const dpr = window.devicePixelRatio || 1;
		const w = canvas.clientWidth;
		const h = canvas.clientHeight;
		canvas.width = w * dpr;
		canvas.height = h * dpr;
		ctx.scale(dpr, dpr);
		ctx.clearRect(0, 0, w, h);

		if (allSeries.length === 0 || allSeries.every(s => s.data.length < 2)) {
			return;
		}

		const pad = { top: 12, right: 12, bottom: 32, left: 52 };
		const plotW = w - pad.left - pad.right;
		const plotH = h - pad.top - pad.bottom;

		let allSteps: number[] = [];
		let allLosses: number[] = [];
		for (const s of allSeries) {
			for (const p of s.data) {
				allSteps.push(p.step);
				if (isFinite(p.loss)) allLosses.push(p.loss);
			}
		}

		const minStep = Math.min(...allSteps);
		const maxStep = Math.max(...allSteps);
		const minLoss = Math.min(...allLosses);
		const maxLoss = Math.max(...allLosses);
		const lossRange = maxLoss - minLoss || 1;
		const stepRange = maxStep - minStep || 1;

		const xScale = (step: number) => pad.left + ((step - minStep) / stepRange) * plotW;
		const yScale = (loss: number) => pad.top + (1 - (loss - minLoss) / lossRange) * plotH;

		// Gridlines
		ctx.textAlign = 'right';
		ctx.textBaseline = 'middle';
		ctx.font = '10px monospace';

		const yTicks = niceSteps(minLoss, maxLoss, 5);
		for (const v of yTicks) {
			const y = yScale(v);
			if (y < pad.top - 1 || y > h - pad.bottom + 1) continue;
			ctx.strokeStyle = '#1f2937';
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(pad.left, y);
			ctx.lineTo(w - pad.right, y);
			ctx.stroke();
			ctx.fillStyle = '#6b7280';
			ctx.fillText(v.toFixed(2), pad.left - 6, y);
		}

		const xTicks = niceSteps(minStep, maxStep, 5);
		ctx.textAlign = 'center';
		ctx.textBaseline = 'top';
		for (const v of xTicks) {
			const x = xScale(v);
			if (x < pad.left - 1 || x > w - pad.right + 1) continue;
			ctx.strokeStyle = '#1f2937';
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(x, pad.top);
			ctx.lineTo(x, h - pad.bottom);
			ctx.stroke();
			ctx.fillStyle = '#6b7280';
			ctx.fillText(v % 1 === 0 ? String(v) : v.toFixed(1), x, h - pad.bottom + 6);
		}

		// Axes
		ctx.strokeStyle = '#374151';
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(pad.left, pad.top);
		ctx.lineTo(pad.left, h - pad.bottom);
		ctx.lineTo(w - pad.right, h - pad.bottom);
		ctx.stroke();

		// Axis label
		ctx.fillStyle = '#6b7280';
		ctx.textAlign = 'center';
		ctx.fillText(t.chart.xAxis, pad.left + plotW / 2, h - 3);

		// Draw non-highlighted series first, then highlighted on top
		const sorted = [...allSeries].sort((a, b) => (a.highlight ? 1 : 0) - (b.highlight ? 1 : 0));
		for (const s of sorted) {
			if (s.data.length < 2) continue;
			const isCurrent = s.label === 'current';
			const isHighlight = s.highlight;
			ctx.strokeStyle = s.color;
			ctx.lineWidth = isCurrent ? 1.5 : isHighlight ? 2 : 1;
			ctx.globalAlpha = isCurrent ? 1 : isHighlight ? 1 : 0.25;
			ctx.beginPath();
			for (let i = 0; i < s.data.length; i++) {
				const x = xScale(s.data[i].step);
				const y = yScale(s.data[i].loss);
				if (i === 0) ctx.moveTo(x, y);
				else ctx.lineTo(x, y);
			}
			ctx.stroke();
			ctx.globalAlpha = 1;
		}
	});
</script>

<div class="relative w-full h-full">
	{#if data.length < 2 && pastRuns.length === 0}
		<div class="absolute inset-0 flex items-center justify-center text-gray-500 text-sm font-mono">
			{t.chart.waiting}
		</div>
	{/if}
	<canvas bind:this={canvas} class="w-full h-full"></canvas>
</div>
