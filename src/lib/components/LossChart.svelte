<script lang="ts">
	type Point = { step: number; loss: number };
	type Series = { data: Point[]; color: string; label?: string };

	let { data, pastRuns = [] }: { data: Point[]; pastRuns?: Series[] } = $props();

	let canvas: HTMLCanvasElement;

	const COLORS = ['#6b7280', '#4b5563', '#374151', '#9ca3af', '#6b7280'];

	$effect(() => {
		if (!canvas) return;

		const allSeries: Series[] = [
			...pastRuns.map((r, i) => ({ ...r, color: r.color || COLORS[i % COLORS.length] })),
			...(data.length >= 2 ? [{ data, color: '#3b82f6', label: 'current' }] : [])
		];

		if (allSeries.length === 0 || allSeries.every(s => s.data.length < 2)) {
			const ctx = canvas.getContext('2d')!;
			const dpr = window.devicePixelRatio || 1;
			canvas.width = canvas.clientWidth * dpr;
			canvas.height = canvas.clientHeight * dpr;
			ctx.scale(dpr, dpr);
			ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
			return;
		}

		const ctx = canvas.getContext('2d')!;
		const dpr = window.devicePixelRatio || 1;
		const w = canvas.clientWidth;
		const h = canvas.clientHeight;
		canvas.width = w * dpr;
		canvas.height = h * dpr;
		ctx.scale(dpr, dpr);

		const pad = { top: 10, right: 10, bottom: 30, left: 50 };
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

		const xScale = (step: number) => pad.left + ((step - minStep) / (maxStep - minStep || 1)) * plotW;
		const yScale = (loss: number) => pad.top + (1 - (loss - minLoss) / lossRange) * plotH;

		ctx.clearRect(0, 0, w, h);

		// Axes
		ctx.strokeStyle = '#374151';
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(pad.left, pad.top);
		ctx.lineTo(pad.left, h - pad.bottom);
		ctx.lineTo(w - pad.right, h - pad.bottom);
		ctx.stroke();

		// Axis labels
		ctx.fillStyle = '#9ca3af';
		ctx.font = '10px monospace';
		ctx.textAlign = 'center';
		ctx.fillText(`${maxStep}`, w - pad.right, h - pad.bottom + 15);
		ctx.fillText('step', pad.left + plotW / 2, h - 5);

		ctx.textAlign = 'right';
		ctx.fillText(maxLoss.toFixed(2), pad.left - 5, pad.top + 10);
		ctx.fillText(minLoss.toFixed(2), pad.left - 5, h - pad.bottom);

		// Draw each series
		for (const s of allSeries) {
			if (s.data.length < 2) continue;
			ctx.strokeStyle = s.color;
			ctx.lineWidth = s.label === 'current' ? 1.5 : 1;
			ctx.globalAlpha = s.label === 'current' ? 1 : 0.4;
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
			waiting for data...
		</div>
	{/if}
	<canvas bind:this={canvas} class="w-full h-full"></canvas>
</div>
