"""Small dependency-free statistical helpers for the fixed three-seed audit."""

from __future__ import annotations

import math

import numpy as np


T_975_DF2 = 4.302652729696142


def paired_three_seed_summary(values) -> dict[str, float | int | list[float]]:
    """Summarize three paired improvements with a two-sided Student-t interval."""
    value = np.asarray(values, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("the V2.16 paired summary requires three finite values")
    mean = float(value.mean())
    standard_deviation = float(value.std(ddof=1))
    half_width = T_975_DF2 * standard_deviation / math.sqrt(3.0)
    return {
        "values_db": value.tolist(),
        "mean_db": mean,
        "sample_standard_deviation_db": standard_deviation,
        "ci95_student_t_db": [mean - half_width, mean + half_width],
        "paired_effect_size_dz": mean / standard_deviation if standard_deviation else 0.0,
        "positive_count": int(np.sum(value > 0)),
    }
