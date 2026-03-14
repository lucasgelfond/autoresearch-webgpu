<script lang="ts">
	type Point = { step: number; loss: number };

	let { data }: { data: Point[] } = $props();

	let canvas: HTMLCanvasElement;

	$effect(() => {
		if (!canvas || data.length < 2) return;

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

		const minStep = data[0].step;
		const maxStep = data[data.length - 1].step;
		const losses = data.map((d) => d.loss).filter((l) => isFinite(l));
		const minLoss = Math.min(...losses);
		const maxLoss = Math.max(...losses);
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

		// Loss line
		ctx.strokeStyle = '#3b82f6';
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		for (let i = 0; i < data.length; i++) {
			const x = xScale(data[i].step);
			const y = yScale(data[i].loss);
			if (i === 0) ctx.moveTo(x, y);
			else ctx.lineTo(x, y);
		}
		ctx.stroke();
	});
</script>

<div class="relative w-full h-full">
	{#if data.length < 2}
		<div class="absolute inset-0 flex items-center justify-center text-gray-500 text-sm font-mono">
			waiting for data...
		</div>
	{/if}
	<canvas bind:this={canvas} class="w-full h-full"></canvas>
</div>
