/** Linear warmup → constant → linear cooldown. Returns a multiplier in [0, 1]. */
export function lrMultiplier(
	progress: number,
	warmupRatio: number,
	cooldownRatio: number
): number {
	if (progress < warmupRatio) {
		return warmupRatio > 0 ? progress / warmupRatio : 1;
	}
	if (progress > 1 - cooldownRatio) {
		return cooldownRatio > 0 ? (1 - progress) / cooldownRatio : 0;
	}
	return 1;
}
