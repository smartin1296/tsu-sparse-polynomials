#!/usr/bin/env python3
"""Subset-guided THRML lab for the XOR dataset."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Dict, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from thermolab.experiments.xor.subset_scan import load_xor_dataset
from thermolab.shared.encoding import spin_encode
from thermolab.shared.interactions import augment_with_interactions
from thermolab.shared.subsets import evaluate_subsets, select_perfect_subsets
from thermolab.shared.thrml import (
    benchmark_polynomial,
    complete_edge_list,
    edge_values_to_dict,
    threshold_weights,
    train_pseudolikelihood,
    unpack_params,
)


@dataclasses.dataclass
class XORThrmlConfig:
    repeats: int = 64
    max_subset: int = 2
    l1_penalty: float = 0.02
    learning_rate: float = 0.08
    train_steps: int = 500
    weight_threshold: float = 0.10
    benchmark_samples: int = 200_000
    benchmark_seed: int = 321


def dense_and_sparse_weights(
    biases: jnp.ndarray,
    edge_values: jnp.ndarray,
    edges: List[Tuple[int, int]],
    threshold: float,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    dense = edge_values_to_dict(edge_values, edges)
    sparse = threshold_weights(dense, threshold)
    return dense, sparse


def evaluate_target_predictions(
    spins: np.ndarray,
    biases: jnp.ndarray,
    weight_matrix: jnp.ndarray,
    target_index: int,
) -> Tuple[List[Tuple[np.ndarray, int, float]], float]:
    unique_rows = np.unique(spins[:, :target_index], axis=0)
    results = []
    for x_bit, y_bit in unique_rows:
        mask = (spins[:, 0] == x_bit) & (spins[:, 1] == y_bit)
        subset = spins[mask]
        target_vals = subset[:, target_index]
        score = biases[target_index] + subset[:, :].dot(weight_matrix[target_index])
        pred_spin = np.where(score >= 0, 1, -1)
        accuracy = float(np.mean(pred_spin == target_vals))
        results.append(((x_bit, y_bit), pred_spin[0], float(score.mean())))
    target_spin = spins[:, target_index]
    full_scores = biases[target_index] + spins.dot(weight_matrix[target_index])
    predictions = np.where(full_scores >= 0, 1, -1)
    full_acc = float(np.mean(predictions == target_spin))
    return results, full_acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--max-subset", type=int, default=2)
    parser.add_argument("--l1", type=float, default=0.02, help="Pseudo-likelihood L1 penalty.")
    parser.add_argument("--lr", type=float, default=0.08, help="Pseudo-likelihood learning rate.")
    parser.add_argument("--steps", type=int, default=500, help="Training steps for pseudo-likelihood.")
    parser.add_argument("--weight-threshold", type=float, default=0.10)
    parser.add_argument("--benchmark-samples", type=int, default=200_000)
    parser.add_argument("--benchmark-seed", type=int, default=321)
    args = parser.parse_args()

    config = XORThrmlConfig(
        repeats=args.repeats,
        max_subset=args.max_subset,
        l1_penalty=args.l1,
        learning_rate=args.lr,
        train_steps=args.steps,
        weight_threshold=args.weight_threshold,
        benchmark_samples=args.benchmark_samples,
        benchmark_seed=args.benchmark_seed,
    )

    data_bits = load_xor_dataset(config)
    data_spins = spin_encode(data_bits)

    subset_results = evaluate_subsets(data_spins, target_index=2, max_subset_size=config.max_subset)
    perfect_subsets = select_perfect_subsets(subset_results)

    print("=== Subset-guided feature discovery ===")
    for res in subset_results:
        is_perfect = tuple(res.features) in perfect_subsets
        subset_tag = "PERFECT" if is_perfect else "----"
        print(
            f"{subset_tag} features={res.features} "
            f"weighted_majority={res.weighted_majority:.3f} weighted_disagreement={res.weighted_disagreement:.3f}"
        )

    augmented_spins, added_subsets = augment_with_interactions(data_spins, perfect_subsets)
    print(f"\nAugmented dataset shape: {augmented_spins.shape}")
    if added_subsets:
        print("Added interaction features for subsets:")
        for subset in added_subsets:
            print(f"  {subset}")
    else:
        print("No additional interaction features added.")

    spins_jax = jnp.asarray(augmented_spins, dtype=jnp.float32)

    class PLConfig:
        train_steps = config.train_steps
        learning_rate = config.learning_rate
        l1_penalty = config.l1_penalty

    edges = complete_edge_list(spins_jax.shape[1])
    trained_params = train_pseudolikelihood(spins_jax, edges, PLConfig)
    biases, edge_vals, weight_matrix = unpack_params(trained_params, spins_jax.shape[1], edges)
    dense_weights, sparse_weights = dense_and_sparse_weights(biases, edge_vals, edges, config.weight_threshold)

    print("\nLearned biases:")
    for idx, bias in enumerate(biases):
        print(f"  bias[{idx}]: {float(bias): .3f}")

    print("\nLearned edge weights (dense):")
    for edge, weight in sorted(dense_weights.items()):
        print(f"  weight{edge}: {weight: .3f}")

    print(f"\nSparse edges above threshold {config.weight_threshold}:")
    if sparse_weights:
        for edge, weight in sorted(sparse_weights.items()):
            print(f"  {edge}: {weight: .3f}")
    else:
        print("  (none)")

    # Evaluate predictions for target spin (index 2)
    target_index = 2
    pattern_stats, accuracy = evaluate_target_predictions(augmented_spins, biases, weight_matrix, target_index)
    print(f"\nDeterministic prediction accuracy on observed data: {accuracy:.3f}")
    for (x_bit, y_bit), pred, score in pattern_stats:
        print(f"  (X={x_bit}, Y={y_bit}) -> predicted Z spin {pred:+d}, avg score={score:.3f}")

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
