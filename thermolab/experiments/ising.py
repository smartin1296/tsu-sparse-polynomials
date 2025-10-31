#!/usr/bin/env python3
"""THRML toy lab for sparse polynomial exploration."""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.pgm import SpinNode

from thermolab.shared.thrml import (
    benchmark_polynomial,
    complete_edge_list,
    edge_values_to_dict,
    threshold_weights,
    train_pseudolikelihood,
    unpack_params,
)

EdgeKey = Tuple[int, int]


@dataclasses.dataclass
class IsingLabConfig:
    """Parameters for the toy Ising experiment."""

    seed: int = 7
    biases: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.25)
    edge_weights: Dict[EdgeKey, float] = dataclasses.field(
        default_factory=lambda: {
            (0, 1): 0.9,
            (1, 2): -0.6,
            (2, 3): 0.4,
            (0, 3): 0.3,
        }
    )
    beta: float = 1.0
    n_warmup: int = 512
    n_samples: int = 2048
    steps_per_sample: int = 4
    correlation_threshold: float = 0.20
    weight_threshold: float = 0.10
    train_steps: int = 600
    learning_rate: float = 0.05
    l1_penalty: float = 0.01
    benchmark_samples: int = 200_000
    benchmark_seed: int = 123


def _sorted_edges(weights: Dict[EdgeKey, float]) -> Iterable[EdgeKey]:
    return sorted(weights.keys(), key=lambda edge: (edge[0], edge[1]))


def build_model(config: IsingLabConfig) -> tuple[list[SpinNode], IsingEBM, IsingSamplingProgram]:
    nodes = [SpinNode() for _ in range(len(config.biases))]
    edge_index = list(_sorted_edges(config.edge_weights))
    edges = [(nodes[i], nodes[j]) for (i, j) in edge_index]
    weights = jnp.array([config.edge_weights[(i, j)] for (i, j) in edge_index], dtype=jnp.float32)
    biases = jnp.array(config.biases, dtype=jnp.float32)
    beta = jnp.array(config.beta, dtype=jnp.float32)

    ebm = IsingEBM(nodes, edges, biases, weights, beta)

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(ebm, free_blocks, [])
    return nodes, ebm, program


def spins_from_samples(samples: jnp.ndarray) -> jnp.ndarray:
    """Convert boolean spin states to {-1, 1} representation."""
    return 2 * samples.astype(jnp.int8) - 1


def compute_pair_correlations(spins: jnp.ndarray) -> Dict[EdgeKey, float]:
    n_nodes = spins.shape[1]
    corr: Dict[EdgeKey, float] = {}
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            corr[(i, j)] = float(jnp.mean(spins[:, i] * spins[:, j]))
    return corr


def threshold_correlations(corr: Dict[EdgeKey, float], threshold: float) -> Dict[EdgeKey, float]:
    return {edge: value for edge, value in corr.items() if abs(value) >= threshold}


def main() -> None:
    config = IsingLabConfig()
    nodes, ebm, program = build_model(config)
    edge_index = list(_sorted_edges(config.edge_weights))
    train_edges = complete_edge_list(len(nodes))

    key = jax.random.key(config.seed)
    key_init, key_sample = jax.random.split(key)

    init_state = hinton_init(key_init, ebm, program.gibbs_spec.free_blocks, ())
    schedule = SamplingSchedule(
        n_warmup=config.n_warmup,
        n_samples=config.n_samples,
        steps_per_sample=config.steps_per_sample,
    )

    sampled = sample_states(
        key_sample,
        program,
        schedule,
        init_state,
        [],
        [Block(nodes)],
    )[0]

    spins = spins_from_samples(sampled)
    magnetisation = jnp.mean(spins, axis=0)
    correlations = compute_pair_correlations(spins)
    active_corr = threshold_correlations(correlations, config.correlation_threshold)

    print("=== THRML Ising lab ===")
    print(f"Samples collected: {config.n_samples}")
    print(f"Magnetisation (avg spin): {magnetisation}")
    print("\nPair correlations (empirical):")
    for edge, value in sorted(correlations.items()):
        print(f"  {edge}: {value:.3f}")

    print(f"\nCorrelations above threshold {config.correlation_threshold}:")
    for edge, value in sorted(active_corr.items()):
        print(f"  {edge}: {value:.3f}")

    # Train pseudo-likelihood with L1 sparsity
    trained_params = train_pseudolikelihood(spins, train_edges, config)
    learned_biases, learned_edge_vals, learned_weight_mat = unpack_params(trained_params, len(nodes), train_edges)
    learned_edges = edge_values_to_dict(learned_edge_vals, train_edges)
    sparse_edges = threshold_weights(learned_edges, config.weight_threshold)

    print("\nLearned parameters (pseudo-likelihood with L1):")
    for idx, bias in enumerate(learned_biases):
        print(f"  bias[{idx}]: {float(bias): .3f}")
    for edge, weight in sorted(learned_edges.items()):
        print(f"  weight{edge}: {weight: .3f}")

    print(f"\nWeights above threshold {config.weight_threshold}:")
    if sparse_edges:
        for edge, weight in sorted(sparse_edges.items()):
            print(f"  {edge}: {weight: .3f}")
    else:
        print("  (none)")

    target = {
        (i, j): float(w)
        for (i, j), w in zip(
            _sorted_edges(config.edge_weights),
            ebm.weights.tolist(),
        )
    }
    print("\nConfigured edge weights (ground truth):")
    for edge, weight in target.items():
        print(f"  {edge}: {weight: .3f}")

    # Benchmark deterministic polynomial evaluation
    dense_time, sparse_time = benchmark_polynomial(
        biases=np.asarray(learned_biases),
        dense_weights=learned_edges,
        sparse_weights=sparse_edges if sparse_edges else {},
        num_samples=config.benchmark_samples,
        seed=config.benchmark_seed,
    )
    multiplier = dense_time / sparse_time if sparse_time > 0 else float("inf")

    print(
        f"\nDeterministic polynomial evaluation over {config.benchmark_samples} samples:\n"
        f"  dense time:  {dense_time:.4f}s\n"
        f"  sparse time: {sparse_time:.4f}s\n"
        f"  speedup:     {multiplier:.2f}×"
    )


if __name__ == "__main__":
    main()
