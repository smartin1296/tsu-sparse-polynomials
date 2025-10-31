#!/usr/bin/env python3
"""Utilities for analysing label consistency across feature subsets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class GroupSummary:
    key: Tuple[int, ...]
    counts: Dict[int, int]
    total: int
    majority_fraction: float
    disagreement: float


@dataclass
class SubsetResult:
    features: Tuple[int, ...]
    weighted_majority: float
    weighted_disagreement: float
    groups: List[GroupSummary]


def _group_targets(
    data: np.ndarray,
    feature_indices: Sequence[int],
    target_index: int,
) -> Dict[Tuple[int, ...], List[int]]:
    grouped: Dict[Tuple[int, ...], List[int]] = {}
    for row in data:
        key = tuple(int(row[idx]) for idx in feature_indices)
        grouped.setdefault(key, []).append(int(row[target_index]))
    return grouped


def _disagreement_from_counts(counts: Dict[int, int], total: int) -> float:
    if total == 0:
        return 0.0
    max_count = max(counts.values())
    majority_fraction = max_count / total
    return 1.0 - majority_fraction


def evaluate_subset(
    data: np.ndarray,
    feature_indices: Sequence[int],
    target_index: int,
) -> SubsetResult:
    grouped = _group_targets(data, feature_indices, target_index)
    total_samples = len(data)

    summaries: List[GroupSummary] = []
    weighted_majority = 0.0
    weighted_disagreement = 0.0

    for key, targets in grouped.items():
        counter = Counter(targets)
        total = len(targets)
        majority_fraction = max(counter.values()) / total
        disagreement = _disagreement_from_counts(counter, total)
        weight = total / total_samples
        weighted_majority += weight * majority_fraction
        weighted_disagreement += weight * disagreement
        summaries.append(
            GroupSummary(
                key=key,
                counts=dict(counter),
                total=total,
                majority_fraction=majority_fraction,
                disagreement=disagreement,
            )
        )

    return SubsetResult(
        features=tuple(feature_indices),
        weighted_majority=weighted_majority,
        weighted_disagreement=weighted_disagreement,
        groups=summaries,
    )


def evaluate_subsets(
    data: np.ndarray,
    target_index: int,
    max_subset_size: int,
) -> List[SubsetResult]:
    n_features = data.shape[1] - 1
    feature_indices = [idx for idx in range(data.shape[1]) if idx != target_index]

    results: List[SubsetResult] = []
    for size in range(1, max_subset_size + 1):
        for subset in combinations(feature_indices, size):
            results.append(evaluate_subset(data, subset, target_index))
    return results


def select_perfect_subsets(
    results: Sequence[SubsetResult],
    *,
    tolerance: float = 1e-9,
) -> List[Tuple[int, ...]]:
    """Pick minimal subsets with zero disagreement (within tolerance)."""
    sorted_results = sorted(results, key=lambda r: (len(r.features), r.features))
    selected: List[Tuple[int, ...]] = []
    for res in sorted_results:
        if res.weighted_disagreement > tolerance:
            continue
        feats = tuple(res.features)
        if any(set(sel).issubset(feats) and set(sel) != set(feats) for sel in selected):
            continue
        selected.append(feats)
    return selected


def rank_singletons_by_majority(
    results: Iterable[SubsetResult],
) -> List[Tuple[int, float]]:
    singles = [res for res in results if len(res.features) == 1]
    singles.sort(key=lambda r: r.weighted_majority, reverse=True)
    return [(int(res.features[0]), res.weighted_majority) for res in singles]


def format_result(res: SubsetResult) -> str:
    lines = [
        f"Features {res.features}: weighted majority={res.weighted_majority:.3f}, weighted disagreement={res.weighted_disagreement:.3f}"
    ]
    for group in res.groups:
        counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(group.counts.items()))
        lines.append(
            f"  key={group.key} total={group.total:>3} majority={group.majority_fraction:.3f} disagreement={group.disagreement:.3f} [{counts_str}]"
        )
    return "\n".join(lines)
