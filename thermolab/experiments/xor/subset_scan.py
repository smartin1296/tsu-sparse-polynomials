#!/usr/bin/env python3
"""Inspect XOR subset consistency and connect to the THRML experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from thermolab.shared.datasets import load_csv_dataset
from thermolab.shared.encoding import spin_encode
from thermolab.shared.subsets import evaluate_subsets, format_result


@dataclass
class XORDatasetConfig:
    repeats: int = 64


def load_xor_dataset(config: XORDatasetConfig) -> np.ndarray:
    values, _ = load_csv_dataset("xor.csv")
    base_rows = values.astype(int)
    if config.repeats <= 1:
        return base_rows
    return np.repeat(base_rows, config.repeats, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=64, help="Number of times to repeat each XOR pattern.")
    parser.add_argument(
        "--max_subset",
        type=int,
        default=2,
        help="Largest feature subset size to inspect.",
    )
    args = parser.parse_args()

    config = XORDatasetConfig(repeats=args.repeats)
    data_bits = load_xor_dataset(config)
    data_spins = spin_encode(data_bits)

    print("=== XOR subset consistency (bits) ===")
    for result in evaluate_subsets(data_bits, target_index=2, max_subset_size=args.max_subset):
        print(format_result(result))

    print("\n=== XOR subset consistency (spin encoding) ===")
    for result in evaluate_subsets(data_spins, target_index=2, max_subset_size=args.max_subset):
        print(format_result(result))

    print(
        "\nObservation: single features (X or Y) remain ambiguous (majority=0.5), "
        "while the joint pair (X,Y) becomes perfectly consistent (majority=1.0). "
        "This indicates the need for at least a two-variable interaction when modelling XOR."
    )


if __name__ == "__main__":
    main()
