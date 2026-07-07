"""Parameter space definition and Latin Hypercube Sampling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParamRange:
    """One optimizable parameter."""

    name: str
    low: float
    high: float
    step: float = 0.0  # 0 = continuous
    is_int: bool = False
    symbol: str = ""  # "" = global, "QQQ" = per-symbol

    def snap(self, value: float) -> float:
        """Snap value to grid if step > 0."""
        if self.step > 0:
            value = round(value / self.step) * self.step
        if self.is_int:
            value = round(value)
        return value


def latin_hypercube_sample(
    param_space: list[ParamRange],
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Generate n_samples using Latin Hypercube Sampling.

    Returns a list of dicts, each mapping param name to sampled value.
    """
    rng = np.random.default_rng(seed)
    n_params = len(param_space)

    # LHS: divide each dimension into n_samples equal intervals
    # Sample one point per interval, then shuffle columns
    samples = np.zeros((n_samples, n_params))

    for j, p in enumerate(param_space):
        # Create stratified intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Sample uniformly within each interval
        points = rng.uniform(intervals[:-1], intervals[1:])
        # Shuffle
        rng.shuffle(points)
        # Scale to parameter range
        samples[:, j] = p.low + points * (p.high - p.low)

    # Snap to grid
    result = []
    for i in range(n_samples):
        sample = {}
        for j, p in enumerate(param_space):
            sample[p.name] = p.snap(samples[i, j])
        result.append(sample)

    return result
