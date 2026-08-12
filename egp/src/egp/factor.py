"""Spectral factorization: autocovariance <-> minimum-phase FIR taps.

Implements the cepstral (Kolmogorov) construction of Section 4.2 of the
paper: given an autocovariance gamma, find FIR taps h with
sum_j h_j h_{j+k} = gamma_k and all roots inside the unit circle, so that h_0
equals the one-step prediction standard deviation (Szego).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len


def autocovariance_from_taps(h, max_lag=None) -> np.ndarray:
    """gamma_k = sum_j h_j h_{j+k} for k = 0 .. max_lag."""
    h = np.asarray(h, dtype=float).ravel()
    n = h.size
    if max_lag is None:
        max_lag = n - 1
    nfft = next_fast_len(2 * n + 1)
    spec = np.fft.rfft(h, nfft)
    gamma = np.fft.irfft(spec * np.conj(spec), nfft)[: max_lag + 1]
    return np.ascontiguousarray(gamma.real)


def spectrum_from_autocov(gamma, nfft) -> np.ndarray:
    """Power spectrum on the full FFT grid: S_m = gamma_0 + 2 sum_k gamma_k cos(w_m k)."""
    gamma = np.asarray(gamma, dtype=float).ravel()
    k = gamma.size - 1
    if 2 * k + 1 > nfft:
        raise ValueError(f"nfft={nfft} too small for {k + 1} autocovariance lags")
    circ = np.zeros(nfft)
    circ[: k + 1] = gamma
    if k > 0:
        circ[nfft - k :] = gamma[:0:-1]
    return np.fft.fft(circ).real


@dataclass
class Factorization:
    """Minimum-phase FIR taps together with quality diagnostics."""

    taps: np.ndarray
    floored_fraction: float
    autocov_error: float  # max |gamma^h - gamma| / gamma_0 over the target lags
    nfft: int

    @property
    def h0(self) -> float:
        return float(self.taps[0])


def minimum_phase_from_autocov(
    gamma, n_taps: int, nfft: int | None = None, floor: float = 1e-14
) -> Factorization:
    """Minimum-phase spectral factor of an autocovariance sequence.

    Parameters
    ----------
    gamma : array
        Autocovariance at lags 0, 1, ..., K.
    n_taps : int
        Number of FIR taps L to keep.
    nfft : int, optional
        FFT grid size; defaults to a fast length >= max(32 L, 32 (K+1), 2**18).
        A generous grid matters: the log-spectrum has integrable singularities
        wherever S vanishes, and only a fine grid keeps their contribution from
        being dominated by ``floor``.
    floor : float
        The spectrum is floored at ``floor * max(S)`` before taking logs.  This
        regularizes spectral nulls; ``floored_fraction`` reports how much of the
        grid it touched, which is a warning sign when large.  Raising it above
        the true stopband power of a sharp filter biases ``h0`` upward (for a
        65-tap Hamming lowpass, ``floor=1e-8`` overstates ``h0`` by ~27%).
    """
    gamma = np.asarray(gamma, dtype=float).ravel()
    if gamma[0] <= 0:
        raise ValueError("gamma[0] must be positive")
    n_taps = int(n_taps)
    if n_taps < 1:
        raise ValueError("n_taps must be >= 1")
    if nfft is None:
        nfft = next_fast_len(max(32 * n_taps, 32 * gamma.size, 1 << 18))
    nfft = int(nfft)

    spec = spectrum_from_autocov(gamma, nfft)
    smax = spec.max()
    if smax <= 0:
        raise ValueError("autocovariance has a non-positive spectrum")
    thresh = floor * smax
    floored_fraction = float(np.mean(spec < thresh))
    spec = np.maximum(spec, thresh)

    # Real cepstrum of sqrt(S), folded to its causal part, then exponentiated.
    ceps = np.fft.ifft(0.5 * np.log(spec)).real
    causal = np.zeros(nfft)
    half = nfft // 2
    causal[0] = ceps[0]
    causal[1:half] = 2.0 * ceps[1:half]
    causal[half] = ceps[half]
    taps_full = np.fft.ifft(np.exp(np.fft.fft(causal))).real
    taps = np.ascontiguousarray(taps_full[:n_taps])
    if taps[0] < 0:
        taps = -taps
    if not np.isfinite(taps).all() or taps[0] <= 0:
        raise RuntimeError("spectral factorization failed; try a larger floor or nfft")

    achieved = autocovariance_from_taps(taps, max_lag=gamma.size - 1)
    autocov_error = float(np.max(np.abs(achieved - gamma)) / gamma[0])
    return Factorization(taps, floored_fraction, autocov_error, nfft)


def minimum_phase_from_taps(
    kernel, n_taps: int | None = None, nfft: int | None = None, floor: float = 1e-14
) -> Factorization:
    """Minimum-phase kernel with the same autocovariance as ``kernel``.

    Any convolution kernel defines the same Gaussian process law as its
    minimum-phase counterpart, but only the minimum-phase version has a usable
    leading tap (see Section 4.2 of the paper).  Linear-phase designs such as
    ``firwin`` output must be passed through this.
    """
    kernel = np.asarray(kernel, dtype=float).ravel()
    if n_taps is None:
        n_taps = kernel.size
    gamma = autocovariance_from_taps(kernel, max_lag=kernel.size - 1)
    return minimum_phase_from_autocov(gamma, n_taps=n_taps, nfft=nfft, floor=floor)


def szego_h0(gamma, nfft: int | None = None, floor: float = 1e-14) -> float:
    """One-step prediction standard deviation exp(mean(log S) / 2)."""
    gamma = np.asarray(gamma, dtype=float).ravel()
    if nfft is None:
        nfft = next_fast_len(max(32 * gamma.size, 1 << 18))
    spec = spectrum_from_autocov(gamma, nfft)
    spec = np.maximum(spec, floor * spec.max())
    return float(np.exp(0.5 * np.mean(np.log(spec))))
