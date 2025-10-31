#!/usr/bin/env python3
"""Encoding helpers for spin-based polynomial models."""

from __future__ import annotations

import numpy as np


def spin_encode(bits: np.ndarray) -> np.ndarray:
    """Map binary {0,1} data to spin {-1,1} representation."""
    return 2 * bits - 1

