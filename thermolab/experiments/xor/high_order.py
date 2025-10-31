#!/usr/bin/env python3
"""Demonstrate XOR modelling with a higher-order spin factor in THRML."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from typing import List

from thrml import Block, SamplingSchedule, sample_states
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.models.ebm import FactorizedEBM
from thrml.pgm import SpinNode


class TripleSpinEBM(FactorizedEBM):
    """Factorised EBM whose energy contains a three-spin interaction."""

    blocks: List[Block]

    def __init__(self, weight: float):
        blocks = [Block([SpinNode()]) for _ in range(3)]
        weights = jnp.array([weight], dtype=jnp.float32)
        factor = SpinEBMFactor(blocks, weights)
        super().__init__([factor])
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "factor", factor)


def percentage_xor(samples: jnp.ndarray) -> float:
    spins = 2 * samples.astype(jnp.int8) - 1
    prod = spins[:, 0] * spins[:, 1] * spins[:, 2]
    # XOR is satisfied when s_x * s_y * s_z = -1
    satisfied = jnp.mean((prod == -1).astype(jnp.float32))
    return float(satisfied)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=-2.0, help="Interaction weight for s_x s_y s_z term.")
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    model = TripleSpinEBM(args.weight)

    spec = BlockGibbsSpec(
        free_super_blocks=model.blocks,
        clamped_blocks=[],
        node_shape_dtypes=model.node_shape_dtypes,
    )

    samplers = [SpinGibbsConditional() for _ in spec.free_blocks]
    program = FactorSamplingProgram(spec, samplers, model.factors, [])

    schedule = SamplingSchedule(
        n_warmup=args.warmup,
        n_samples=args.samples,
        steps_per_sample=args.steps,
    )

    key = jax.random.PRNGKey(0)
    key_state, key_sample = jax.random.split(key)
    init_keys = jax.random.split(key_state, len(spec.free_blocks))
    init_state = [
        jax.random.bernoulli(k, p=0.5, shape=(1,)).astype(jnp.bool_)
        for k in init_keys
    ]

    all_nodes = [blk.nodes[0] for blk in model.blocks]
    samples = sample_states(key_sample, program, schedule, init_state, [], [Block(all_nodes)])[0]

    xor_frac = percentage_xor(samples)
    print(f"Weight = {args.weight:+.2f}")
    print(f"Fraction of states satisfying XOR parity (s_z = s_x XOR s_y): {xor_frac:.3f}")

    unique, counts = jnp.unique(samples, axis=0, return_counts=True)
    unique = np.array(unique)
    counts = np.array(counts)
    print("Sampled configuration counts:")
    for state, count in zip(unique, counts):
        print(f"  {tuple(bool(x) for x in state)}: {int(count)}")


if __name__ == "__main__":
    main()
