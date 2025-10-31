#!/usr/bin/env python3
"""Subset-guided sparse polynomial exploration on the Breast Cancer dataset."""

from __future__ import annotations

import argparse
import dataclasses
from typing import List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from thermolab.shared.datasets import load_csv_dataset, train_test_split
from thermolab.shared.encoding import spin_encode
from thermolab.shared.interactions import augment_with_interactions
from thermolab.shared.subsets import evaluate_subsets, rank_singletons_by_majority
from thermolab.shared.thrml import (
    benchmark_polynomial,
    complete_edge_list,
    edge_values_to_dict,
    threshold_weights,
    train_pseudolikelihood,
    unpack_params,
)


@dataclasses.dataclass
class BreastConfig:
    max_subset: int = 2
    l1_penalty: float = 0.01
    learning_rate: float = 0.05
    train_steps: int = 800
    weight_threshold: float = 0.08
    benchmark_samples: int = 200_000
    benchmark_seed: int = 4242
    test_size: float = 0.3
    split_seed: int = 321
    threshold_quantile: float = 0.5
    feature_indices: Sequence[int] | None = None
    top_features: int = 8


def binarise_features(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (values >= thresholds).astype(int)


def prepare_splits(config: BreastConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    values, column_names = load_csv_dataset("breast_cancer.csv")
    features = values[:, :-1]
    targets = values[:, -1].astype(int)

    all_feature_names = list(column_names[:-1])

    if config.feature_indices is not None:
        features = features[:, config.feature_indices]
        feature_names = [all_feature_names[idx] for idx in config.feature_indices]
    else:
        feature_names = all_feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=config.test_size,
        seed=config.split_seed,
        stratify=True,
    )

    thresholds = np.quantile(X_train, config.threshold_quantile, axis=0)
    train_bits = binarise_features(X_train, thresholds)
    test_bits = binarise_features(X_test, thresholds)

    return train_bits, test_bits, y_train, y_test, feature_names, thresholds


def describe_subset_results(results, feature_names):
    print("=== Subset consistency summary ===")
    for res in results:
        feats = tuple(res.features)
        labels = [feature_names[idx] for idx in feats]
        print(
            f"features={feats} ({' * '.join(labels)}) "
            f"weighted_majority={res.weighted_majority:.3f} weighted_disagreement={res.weighted_disagreement:.3f}"
        )


def top_feature_indices_by_majority(results, max_features: int) -> Sequence[int]:
    ranked = rank_singletons_by_majority(results)
    return tuple(idx for idx, _ in ranked[:max_features])


def build_pair_subsets(top_indices: Sequence[int]) -> List[Tuple[int, ...]]:
    from itertools import combinations

    return [tuple(pair) for pair in combinations(top_indices, 2)]


def build_feature_names(base: List[str], added_subsets: Sequence[Tuple[int, ...]]) -> List[str]:
    names = list(base)
    for subset in added_subsets:
        if len(subset) < 2:
            continue
        subset_names = [base[idx] for idx in subset]
        names.append("*".join(subset_names))
    names.append("target")
    return names


def evaluate_accuracy(spins: np.ndarray, biases: jnp.ndarray, weight_matrix: jnp.ndarray) -> float:
    scores = biases[-1] + spins @ weight_matrix[-1]
    preds = np.where(scores >= 0, 1, -1)
    return float(np.mean(preds == spins[:, -1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-subset", type=int, default=2)
    parser.add_argument("--l1", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--weight-threshold", type=float, default=0.08)
    parser.add_argument("--benchmark-samples", type=int, default=200_000)
    parser.add_argument("--benchmark-seed", type=int, default=4242)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=321)
    parser.add_argument("--threshold-quantile", type=float, default=0.5)
    parser.add_argument("--feature-indices", type=str, default=None, help="Comma-separated feature indices to use.")
    parser.add_argument("--top-features", type=int, default=8, help="Number of top single features (by majority agreement) to use for pair interactions.")
    args = parser.parse_args()

    feature_indices = None
    if args.feature_indices:
        feature_indices = tuple(int(x.strip()) for x in args.feature_indices.split(",") if x.strip())

    config = BreastConfig(
        max_subset=args.max_subset,
        l1_penalty=args.l1,
        learning_rate=args.lr,
        train_steps=args.steps,
        weight_threshold=args.weight_threshold,
        benchmark_samples=args.benchmark_samples,
        benchmark_seed=args.benchmark_seed,
        test_size=args.test_size,
        split_seed=args.split_seed,
        threshold_quantile=args.threshold_quantile,
        feature_indices=feature_indices,
        top_features=args.top_features,
    )

    train_bits, test_bits, y_train, y_test, feature_names, thresholds = prepare_splits(config)
    train_data = np.hstack([train_bits, y_train[:, None]])
    test_data = np.hstack([test_bits, y_test[:, None]])

    print("Thresholds used for binarisation:")
    for idx, thresh in enumerate(thresholds):
        print(f"  {feature_names[idx]} >= {thresh:.3f} -> 1")

    train_spins = spin_encode(train_data)
    test_spins = spin_encode(test_data)
    target_index = train_bits.shape[1]

    subset_results = evaluate_subsets(train_spins, target_index=target_index, max_subset_size=1)
    top_singles = top_feature_indices_by_majority(subset_results, max_features=config.top_features)
    pair_subsets = build_pair_subsets(top_singles)
    describe_subset_results(subset_results, feature_names)

    train_aug_spins, added_subsets = augment_with_interactions(train_spins, pair_subsets)
    test_aug_spins, _ = augment_with_interactions(test_spins, added_subsets)
    augmented_names = build_feature_names(feature_names, added_subsets)

    print(f"\nAugmented train shape: {train_aug_spins.shape}")
    print(f"Augmented test shape: {test_aug_spins.shape}")
    if added_subsets:
        print("Added interaction features:")
        for subset in added_subsets:
            if len(subset) < 2:
                continue
            labels = [feature_names[idx] for idx in subset]
            print(f"  {subset} -> {' * '.join(labels)}")
    else:
        print("No interaction features added.")

    spins_jax = jnp.asarray(train_aug_spins, dtype=jnp.float32)

    class PLConfig:
        train_steps = config.train_steps
        learning_rate = config.learning_rate
        l1_penalty = config.l1_penalty

    edges = complete_edge_list(spins_jax.shape[1])
    trained_params = train_pseudolikelihood(spins_jax, edges, PLConfig)
    biases, edge_vals, weight_matrix = unpack_params(trained_params, spins_jax.shape[1], edges)
    dense_weights = edge_values_to_dict(edge_vals, edges)
    sparse_weights = threshold_weights(dense_weights, config.weight_threshold)

    print("\nLearned biases (train):")
    for idx, bias in enumerate(biases):
        label = augmented_names[idx] if idx < len(augmented_names) else f"feat{idx}"
        print(f"  bias[{idx}] ({label}): {float(bias): .3f}")

    print("\nTop edge weights (dense):")
    for edge, weight in sorted(dense_weights.items(), key=lambda kv: -abs(kv[1]))[:12]:
        i, j = edge
        name_i = augmented_names[i] if i < len(augmented_names) else f"feat{i}"
        name_j = augmented_names[j] if j < len(augmented_names) else f"feat{j}"
        print(f"  ({i}, {j}) {name_i} ↔ {name_j}: {weight: .3f}")

    print(f"\nSparse edges above threshold {config.weight_threshold}:")
    if sparse_weights:
        for edge, weight in sorted(sparse_weights.items(), key=lambda kv: -abs(kv[1])):
            i, j = edge
            name_i = augmented_names[i] if i < len(augmented_names) else f"feat{i}"
            name_j = augmented_names[j] if j < len(augmented_names) else f"feat{j}"
            print(f"  ({i}, {j}) {name_i} ↔ {name_j}: {weight: .3f}")
    else:
        print("  (none)")

    train_accuracy = evaluate_accuracy(train_aug_spins, biases, weight_matrix)
    test_accuracy = evaluate_accuracy(test_aug_spins, biases, weight_matrix)
    print(f"\nDeterministic accuracy on train split: {train_accuracy:.3f}")
    print(f"Deterministic accuracy on test split:  {test_accuracy:.3f}")

    dense_time, sparse_time = benchmark_polynomial(
        biases=np.asarray(biases),
        dense_weights=dense_weights,
        sparse_weights=sparse_weights,
        num_samples=config.benchmark_samples,
        seed=config.benchmark_seed,
    )
    speedup = dense_time / sparse_time if sparse_time > 0 else float("inf")
    print(
        f"\nDeterministic polynomial benchmarking ({config.benchmark_samples} samples):\n"
        f"  dense time:  {dense_time:.4f}s\n"
        f"  sparse time: {sparse_time:.4f}s\n"
        f"  speedup:     {speedup:.2f}×"
    )


if __name__ == "__main__":
    main()
