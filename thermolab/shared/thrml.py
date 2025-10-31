#!/usr/bin/env python3
"""Shared THRML utilities for pseudo-likelihood training and benchmarking."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import time

EdgeKey = Tuple[int, int]


def complete_edge_list(n_nodes: int) -> List[EdgeKey]:
    return [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]


def vector_to_weight_matrix(edge_values: jnp.ndarray, n_nodes: int, edges: List[EdgeKey]) -> jnp.ndarray:
    weight_mat = jnp.zeros((n_nodes, n_nodes))
    for idx, (i, j) in enumerate(edges):
        weight_mat = weight_mat.at[i, j].set(edge_values[idx])
        weight_mat = weight_mat.at[j, i].set(edge_values[idx])
    return weight_mat


def unpack_params(params: jnp.ndarray, n_nodes: int, edges: List[EdgeKey]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    biases = params[:n_nodes]
    edge_vals = params[n_nodes:]
    weight_mat = vector_to_weight_matrix(edge_vals, n_nodes, edges)
    return biases, edge_vals, weight_mat


def pseudo_likelihood_loss(
    params: jnp.ndarray,
    spins: jnp.ndarray,
    n_nodes: int,
    edges: List[EdgeKey],
    l1_penalty: float,
) -> jnp.ndarray:
    biases, edge_vals, weight_mat = unpack_params(params, n_nodes, edges)
    fields = biases + spins @ weight_mat
    logits = -2.0 * spins * fields
    data_loss = jnp.mean(jax.nn.softplus(logits))
    penalty = l1_penalty * jnp.sum(jnp.abs(edge_vals))
    return data_loss + penalty


def train_pseudolikelihood(
    spins: jnp.ndarray,
    edges: List[EdgeKey],
    config,
) -> jnp.ndarray:
    """Train pseudo-likelihood parameters with optional L1 penalty."""
    n_nodes = spins.shape[1]
    params = jnp.zeros(n_nodes + len(edges))
    loss_fn = lambda p: pseudo_likelihood_loss(p, spins, n_nodes, edges, config.l1_penalty)
    grad_fn = jax.grad(loss_fn)

    def body_fun(_, current):
        grads = grad_fn(current)
        return current - config.learning_rate * grads

    @jax.jit
    def run_training(initial_params: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.fori_loop(0, config.train_steps, body_fun, initial_params)

    params = run_training(params)
    final_loss = float(loss_fn(params))
    print(f"[train] completed {config.train_steps} steps  loss={final_loss:.4f}")
    return params


def edge_values_to_dict(edge_vals: Iterable[float], edges: List[EdgeKey]) -> Dict[EdgeKey, float]:
    return {edge: float(edge_vals[idx]) for idx, edge in enumerate(edges)}


def threshold_weights(edge_vals: Dict[EdgeKey, float], threshold: float) -> Dict[EdgeKey, float]:
    return {edge: weight for edge, weight in edge_vals.items() if abs(weight) >= threshold}


def evaluate_energy_batch(
    spins: np.ndarray,
    biases: np.ndarray,
    edge_weights: Dict[EdgeKey, float],
) -> np.ndarray:
    energy = -spins @ biases
    for (i, j), weight in edge_weights.items():
        energy -= weight * spins[:, i] * spins[:, j]
    return energy


def benchmark_polynomial(
    biases: np.ndarray,
    dense_weights: Dict[EdgeKey, float],
    sparse_weights: Dict[EdgeKey, float],
    num_samples: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=(num_samples, len(biases)))

    start = time.perf_counter()
    _ = evaluate_energy_batch(spins, biases, dense_weights)
    dense_time = time.perf_counter() - start

    start = time.perf_counter()
    _ = evaluate_energy_batch(spins, biases, sparse_weights)
    sparse_time = time.perf_counter() - start

    return dense_time, sparse_time
