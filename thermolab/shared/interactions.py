#!/usr/bin/env python3
"""Feature engineering helpers for interaction-based THRML experiments."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def augment_with_interactions(
    spins: np.ndarray,
    subsets: Sequence[Tuple[int, ...]],
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Append interaction columns for ``subsets`` to the spin matrix.

    Each subset corresponds to indices within the feature portion of ``spins``
    (all columns except the final target column).
    """
    if not subsets:
        return spins, []

    features = spins[:, :-1]
    target_column = spins[:, -1:]
    extra_columns = []
    added_subsets: List[Tuple[int, ...]] = []

    for subset in subsets:
        if len(subset) < 2:
            continue
        interaction = np.prod(features[:, subset], axis=1, dtype=np.int8)
        extra_columns.append(interaction[:, None])
        added_subsets.append(tuple(subset))

    if not extra_columns:
        return spins, []

    augmented = np.concatenate([features] + extra_columns + [target_column], axis=1)
    return augmented, added_subsets

