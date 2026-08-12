from __future__ import annotations

import math

import numpy as np

QUANTIZER_FLOOR = 0.5 * math.log2(2 * math.pi * math.e / 12.0)


def psd(kernel, n: int = 4096, midpoint: bool = False):
    """|H(omega)|^2 on a grid of [0, pi].

    The midpoint grid avoids spectral nulls at rational multiples of pi, where
    log S is integrable but not evaluable.
    """
    h = np.asarray(kernel, dtype=float).ravel()
    size = 2 * n
    if midpoint:
        rotated = h * np.exp(-1j * (np.pi / size) * np.arange(h.size))
        response = np.fft.fft(rotated, size)[:n]
        omega = np.pi * (np.arange(n) + 0.5) / n
    else:
        response = np.fft.rfft(h, size)
        omega = np.pi * np.arange(response.size) / n
    return omega, np.abs(response) ** 2


def innovation_variance(kernel, n: int = 16384) -> float:
    """exp((1/2pi) int log S dw), zero when S vanishes on a set of positive measure."""
    _, spectrum = psd(kernel, n, midpoint=True)
    with np.errstate(divide="ignore"):
        return float(np.exp(np.mean(np.log(spectrum))))


def entropy_rate_approx(kernel, delta: float, n: int = 16384) -> float:
    """(1/4pi) int log2( 2 pi e (S(w) + Delta^2/12) / Delta^2 ) dw, in bits/sample.

    The high-resolution formula with the quantizer noise floor Delta^2/12 added
    to the spectrum, so that spectral zeros stay finite.
    """
    h = np.trim_zeros(np.asarray(kernel, dtype=float).ravel())
    delta = float(delta)
    _, spectrum = psd(h, n, midpoint=True)
    return 0.5 * (
        math.log2(2 * math.pi * math.e)
        + float(np.mean(np.log2(spectrum + delta**2 / 12.0)))
    ) - math.log2(delta)


def snr_integral(kernel, delta: float, n: int = 16384) -> float:
    """The part of `entropy_rate_approx` above the fixed QUANTIZER_FLOOR."""
    h = np.trim_zeros(np.asarray(kernel, dtype=float).ravel())
    _, spectrum = psd(h, n, midpoint=True)
    return 0.5 * float(np.mean(np.log2(1.0 + spectrum / (float(delta) ** 2 / 12.0))))


def entropy_rate_lower_bound(kernel, delta: float) -> float:
    """(1/2) log2(1 + 2 pi e sigma_inf^2 / Delta^2)."""
    variance = innovation_variance(kernel)
    return 0.5 * math.log2(1.0 + 2 * math.pi * math.e * variance / float(delta) ** 2)
