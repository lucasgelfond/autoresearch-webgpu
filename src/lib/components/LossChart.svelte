<script lang="ts">
	type Point = { step: number; loss: number };
	type Series = { data: Point[]; color: string; label?: string; highlight?: boolean };

	let {
		data,
		pastRuns = [],
		seriesMode = 'all',
		yScaleMode = 'fit',
		yMin = null,
		yMax = null
	}: {
		data: Point[];
		pastRuns?: Series[];
		seriesMode?: 'all' | 'focus';
		yScaleMode?: 'fit' | 'trim' | 'manual';
		yMin?: number | null;
		yMax?: number | null;
	} = $props();

	let container: HTMLDivElement;
	let canvas: HTMLCanvasElement;
	let size = $state({ width: 0, height: 0 });

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

	function quantile(sorted: number[], q: number): number {
		if (sorted.length === 0) return 0;
		if (sorted.length === 1) return sorted[0];
		const idx = (sorted.length - 1) * q;
		const lo = Math.floor(idx);
		const hi = Math.ceil(idx);
		const t = idx - lo;
		return sorted[lo] * (1 - t) + sorted[hi] * t;
	}

	function resolveYDomain(losses: number[]) {
		if (losses.length === 0) return null;

		const sorted = [...losses].sort((a, b) => a - b);
		const minLoss = sorted[0];
		const maxLoss = sorted[sorted.length - 1];

		if (yScaleMode === 'manual' && yMin != null && yMax != null && isFinite(yMin) && isFinite(yMax) && yMax > yMin) {
			return { min: yMin, max: yMax };
		}

		let domainMin = minLoss;
		let domainMax = maxLoss;

		if (yScaleMode === 'trim' && sorted.length >= 4) {
			const q1 = quantile(sorted, 0.25);
			const q3 = quantile(sorted, 0.75);
			const iqr = q3 - q1;
			if (iqr > 0) {
				domainMin = Math.max(minLoss, q1 - 1.5 * iqr);
				domainMax = Math.min(maxLoss, q3 + 1.5 * iqr);
			}
		}

		if (!(domainMax > domainMin)) {
			const center = domainMin;
			return { min: center - 0.5, max: center + 0.5 };
		}

		const pad = (domainMax - domainMin) * 0.05;
		return { min: domainMin - pad, max: domainMax + pad };
	}

	function drawSeries(
		ctx: CanvasRenderingContext2D,
		points: Point[],
		xScale: (step: number) => number,
		yScale: (loss: number) => number
	) {
		let drawing = false;
		for (const point of points) {
			if (!isFinite(point.loss)) {
				drawing = false;
				continue;
			}
			const x = xScale(point.step);
			const y = yScale(point.loss);
			if (!drawing) {
				ctx.moveTo(x, y);
				drawing = true;
			} else {
				ctx.lineTo(x, y);
			}
		}
	}

	function updateSize() {
		if (!container) return;
		const next = {
			width: container.clientWidth,
			height: container.clientHeight
		};
		if (next.width !== size.width || next.height !== size.height) {
			size = next;
		}
	}

	$effect(() => {
		if (!container) return;
		updateSize();
		const observer = new ResizeObserver(() => updateSize());
		observer.observe(container);
		return () => observer.disconnect();
	});

	$effect(() => {
		if (!canvas || size.width === 0 || size.height === 0) return;

		const baseSeries: Series[] = [
			...pastRuns.map((r, i) => ({ ...r, color: r.color || COLORS[i % COLORS.length] })),
			...(data.length >= 2 ? [{ data, color: '#3b82f6', label: 'current' }] : [])
		];
		const allSeries =
			seriesMode === 'focus'
				? baseSeries.filter((series) => series.label === 'current' || series.highlight)
				: baseSeries;

		const ctx = canvas.getContext('2d')!;
		const dpr = window.devicePixelRatio || 1;
		const w = size.width;
		const h = size.height;
		canvas.width = w * dpr;
		canvas.height = h * dpr;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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
		const yDomain = resolveYDomain(allLosses);
		if (!yDomain) return;
		const lossRange = yDomain.max - yDomain.min || 1;
		const stepRange = maxStep - minStep || 1;

		const xScale = (step: number) => pad.left + ((step - minStep) / stepRange) * plotW;
		const yScale = (loss: number) => pad.top + (1 - (loss - yDomain.min) / lossRange) * plotH;

		// Gridlines
		ctx.textAlign = 'right';
		ctx.textBaseline = 'middle';
		ctx.font = '10px monospace';

		const yTicks = niceSteps(yDomain.min, yDomain.max, 5);
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
		ctx.fillText('step', pad.left + plotW / 2, h - 3);

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
			drawSeries(ctx, s.data, xScale, yScale);
			ctx.stroke();
			ctx.globalAlpha = 1;
		}
	});
</script>

<div bind:this={container} class="relative w-full h-full">
	{#if data.length < 2 && pastRuns.length === 0}
		<div class="absolute inset-0 flex items-center justify-center text-gray-500 text-sm font-mono">
			waiting for data...
		</div>
	{/if}
	<canvas bind:this={canvas} class="w-full h-full"></canvas>
</div>
