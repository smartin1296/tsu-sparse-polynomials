#!/usr/bin/env python3
"""Utility helpers for loading tabular datasets from the packaged data directory."""

from __future__ import annotations

from importlib.resources import files
from typing import Sequence, Tuple

import numpy as np


def load_csv_dataset(name: str) -> Tuple[np.ndarray, Sequence[str]]:
    """Load a CSV dataset bundled under ``thermolab.data``.

    Returns a tuple ``(values, column_names)`` where ``values`` is a
    two-dimensional ``numpy`` array.
    """
    resource = files("thermolab.data").joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(f"Dataset {name!r} not found in thermolab.data.")

    with resource.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        values = np.loadtxt(handle, delimiter=",")

    column_names = tuple(col.strip() for col in header.split(","))
    if values.ndim == 1:
        values = values[None, :]
    return values, column_names


def _validate_split_inputs(features: np.ndarray, target: np.ndarray) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if target.ndim != 1:
        raise ValueError("target must be a 1D array")
    if features.shape[0] != target.shape[0]:
        raise ValueError("features and target must have matching first dimension")


def train_test_split(
    features: np.ndarray,
    target: np.ndarray,
    *,
    test_size: float,
    seed: int | None = None,
    stratify: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple reimplementation of ``sklearn.model_selection.train_test_split``.

    Supports optional stratification by the target labels.
    """
    _validate_split_inputs(features, target)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]

    if stratify:
        train_indices: list[int] = []
        test_indices: list[int] = []
        unique_labels = np.unique(target)
        for label in unique_labels:
            class_indices = np.flatnonzero(target == label)
            rng.shuffle(class_indices)
            n_test = max(1, int(round(len(class_indices) * test_size)))
            test_indices.extend(class_indices[:n_test])
            train_indices.extend(class_indices[n_test:])
    else:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_test = int(round(n_samples * test_size))
        test_indices = indices[:n_test].tolist()
        train_indices = indices[n_test:].tolist()

    if not train_indices:
        raise ValueError("Not enough samples to create a training split.")
    if not test_indices:
        raise ValueError("Not enough samples to create a test split.")

    train_indices_arr = np.array(train_indices, dtype=int)
    test_indices_arr = np.array(test_indices, dtype=int)
    rng.shuffle(train_indices_arr)
    rng.shuffle(test_indices_arr)

    x_train = features[train_indices_arr]
    x_test = features[test_indices_arr]
    y_train = target[train_indices_arr]
    y_test = target[test_indices_arr]

    return x_train, x_test, y_train, y_test
