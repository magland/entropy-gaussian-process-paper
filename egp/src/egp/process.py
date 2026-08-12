"""Simulation of the process and its uniform quantization."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def quantize(x, delta: float) -> np.ndarray:
    """Round to the nearest multiple of ``delta``; returns integer symbols."""
    return np.floor(np.asarray(x, dtype=float) / float(delta) + 0.5).astype(np.int64)


def dequantize(y, delta: float) -> np.ndarray:
    return np.asarray(y, dtype=float) * float(delta)


def generate(taps, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a stationary sample X_1..X_n of the FIR process.

    Innovations W_{2-L} .. W_n are drawn i.i.d. standard normal so that every
    returned sample has its full history; there is no burn-in transient.
    """
    taps = np.asarray(taps, dtype=float).ravel()
    n_taps = taps.size
    w = rng.standard_normal(n + n_taps - 1)
    if n_taps == 1:
        return taps[0] * w
    return fftconvolve(w, taps, mode="valid")[:n]


def generate_quantized(taps, n: int, delta: float, rng: np.random.Generator):
    """Return ``(x, y)``: the continuous sample and its integer symbols."""
    x = generate(taps, n, rng)
    return x, quantize(x, delta)
