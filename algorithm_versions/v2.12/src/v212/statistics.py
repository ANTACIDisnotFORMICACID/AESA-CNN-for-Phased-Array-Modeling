"""Pre-declared paired statistics for the ten-seed Stage-1/Stage-2 comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np


@dataclass(frozen=True)
class PairedComparison:
    mean_improvement_db: float
    median_improvement_db: float
    confidence_interval_95_db: tuple[float, float]
    two_sided_sign_flip_p: float
    wins: int
    count: int


def paired_comparison(
    baseline_nmse_db: np.ndarray,
    candidate_nmse_db: np.ndarray,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 21212,
) -> PairedComparison:
    baseline = np.asarray(baseline_nmse_db, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_nmse_db, dtype=np.float64).reshape(-1)
    if baseline.shape != candidate.shape or baseline.size < 2:
        raise ValueError("paired arrays must have matching shape and at least two values")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired metrics must be finite")
    improvement = baseline - candidate
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(improvement), size=(bootstrap_samples, len(improvement)))
    bootstrap = improvement[samples].mean(axis=1)
    interval = tuple(float(x) for x in np.quantile(bootstrap, (0.025, 0.975)))
    observed = abs(float(improvement.mean()))
    if len(improvement) <= 20:
        means = np.asarray(
            [np.mean(improvement * np.asarray(signs)) for signs in itertools.product((-1.0, 1.0), repeat=len(improvement))]
        )
    else:
        signs = generator.choice((-1.0, 1.0), size=(100_000, len(improvement)))
        means = (signs * improvement).mean(axis=1)
    p_value = float(np.mean(np.abs(means) >= observed - 1e-15))
    return PairedComparison(
        mean_improvement_db=float(improvement.mean()),
        median_improvement_db=float(np.median(improvement)),
        confidence_interval_95_db=interval,
        two_sided_sign_flip_p=p_value,
        wins=int(np.sum(improvement > 0)),
        count=len(improvement),
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    if any(not 0.0 <= value <= 1.0 for value in p_values.values()):
        raise ValueError("p-values must lie in [0,1]")
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted
