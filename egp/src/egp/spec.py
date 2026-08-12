"""Ways of specifying the Gaussian process, all resolved to minimum-phase taps.

A process may be given as
  * a preset (moving average, lowpass, ...),
  * an arbitrary convolution kernel applied to i.i.d. N(0, 1),
  * an arbitrary autocovariance sequence.

All three are normalized to the same internal representation: a vector of
minimum-phase FIR taps ``h`` such that X_t = sum_j h_j W_{t-j} with W i.i.d.
standard normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import firwin

from .factor import (
    Factorization,
    autocovariance_from_taps,
    minimum_phase_from_autocov,
    minimum_phase_from_taps,
)

PRESETS: dict[str, str] = {
    "white": "white noise (single tap); Y_t are i.i.d.",
    "ma": "moving average / boxcar smoother of length --taps",
    "gaussian": "Gaussian smoothing kernel of width --tau, length --taps",
    "lowpass": "windowed-sinc lowpass, --cutoff (Hz), --fs, --taps",
    "highpass": "windowed-sinc highpass, --cutoff (Hz), --fs, --taps",
    "bandpass": "windowed-sinc bandpass, --band LO HI (Hz), --fs, --taps",
    "ar1": "AR(1) with correlation --rho, truncated to --taps",
    "diff": "first difference [1, -1] (spectral null at DC)",
}

_DEFAULT_TAPS: dict[str, int] = {
    "white": 1,
    "ma": 8,
    "gaussian": 33,
    "lowpass": 65,
    "highpass": 65,
    "bandpass": 65,
    "ar1": 64,
    "diff": 2,
}


@dataclass
class ProcessSpec:
    """A fully resolved stationary Gaussian process in FIR form."""

    taps: np.ndarray  # minimum-phase h, length L
    label: str
    fs: float = 1.0
    design_kernel: np.ndarray | None = None  # user/preset kernel before factorization
    target_autocov: np.ndarray | None = None  # autocovariance the taps aim to match
    autocov_error: float = 0.0
    floored_fraction: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_taps(self) -> int:
        return int(self.taps.size)

    @property
    def h0(self) -> float:
        """One-step prediction standard deviation."""
        return float(self.taps[0])

    @property
    def sigma(self) -> float:
        """Marginal standard deviation of X_t."""
        return float(np.sqrt(np.dot(self.taps, self.taps)))

    def autocovariance(self, max_lag: int | None = None) -> np.ndarray:
        return autocovariance_from_taps(self.taps, max_lag=max_lag)

    def spectrum(self, nfft: int = 4096):
        """One-sided power spectrum of the realized process.

        Returns ``(freq, S)`` with ``freq`` in Hz when ``fs`` was supplied.
        """
        spec = np.fft.rfft(self.taps, nfft)
        power = np.abs(spec) ** 2
        freq = np.fft.rfftfreq(nfft, d=1.0 / self.fs)
        return freq, power


def _rescale(fac: Factorization, sigma: float) -> np.ndarray:
    taps = fac.taps
    current = float(np.sqrt(np.dot(taps, taps)))
    return taps * (sigma / current)


def _finish(
    fac: Factorization,
    label: str,
    *,
    fs: float,
    sigma: float | None,
    design_kernel: np.ndarray | None,
    target_autocov: np.ndarray | None,
    params: dict[str, Any],
) -> ProcessSpec:
    taps = _rescale(fac, sigma) if sigma is not None else fac.taps
    return ProcessSpec(
        taps=np.ascontiguousarray(taps, dtype=float),
        label=label,
        fs=fs,
        design_kernel=design_kernel,
        target_autocov=target_autocov,
        autocov_error=fac.autocov_error,
        floored_fraction=fac.floored_fraction,
        params=params,
    )


def from_taps(
    kernel,
    n_taps: int | None = None,
    *,
    sigma: float | None = 1.0,
    fs: float = 1.0,
    floor: float = 1e-14,
    nfft: int | None = None,
    label: str = "custom kernel",
) -> ProcessSpec:
    """Process defined by convolving i.i.d. N(0, 1) with ``kernel``."""
    kernel = np.asarray(kernel, dtype=float).ravel()
    if kernel.size == 0:
        raise ValueError("kernel is empty")
    fac = minimum_phase_from_taps(kernel, n_taps=n_taps, nfft=nfft, floor=floor)
    return _finish(
        fac,
        label,
        fs=fs,
        sigma=sigma,
        design_kernel=kernel,
        target_autocov=autocovariance_from_taps(kernel),
        params={"n_taps": fac.taps.size},
    )


def from_autocov(
    gamma,
    n_taps: int | None = None,
    *,
    sigma: float | None = 1.0,
    fs: float = 1.0,
    floor: float = 1e-14,
    nfft: int | None = None,
    label: str = "custom autocovariance",
) -> ProcessSpec:
    """Process defined by its autocovariance at lags 0, 1, ..., K."""
    gamma = np.asarray(gamma, dtype=float).ravel()
    if n_taps is None:
        n_taps = 4 * gamma.size
    fac = minimum_phase_from_autocov(gamma, n_taps=n_taps, nfft=nfft, floor=floor)
    return _finish(
        fac,
        label,
        fs=fs,
        sigma=sigma,
        design_kernel=None,
        target_autocov=gamma,
        params={"n_taps": n_taps},
    )


def _design_kernel(name: str, *, taps: int, fs: float, cutoff, band, rho, tau, window):
    """Return (kernel, label, params) for a preset."""
    if name == "white":
        return np.ones(1), "white noise", {}
    if name == "ma":
        return np.ones(taps), f"moving average (width {taps})", {"width": taps}
    if name == "gaussian":
        j = np.arange(taps) - (taps - 1) / 2.0
        return (
            np.exp(-0.5 * (j / tau) ** 2),
            f"Gaussian kernel (tau={tau:g}, {taps} taps)",
            {"tau": tau},
        )
    if name == "diff":
        return np.array([1.0, -1.0]), "first difference", {}
    if name == "ar1":
        if not -1.0 < rho < 1.0:
            raise ValueError("--rho must lie in (-1, 1)")
        j = np.arange(taps)
        return (
            np.sqrt(1.0 - rho**2) * rho**j,
            f"AR(1) (rho={rho:g}, {taps} taps)",
            {"rho": rho},
        )
    if name in ("lowpass", "highpass", "bandpass"):
        if name == "bandpass":
            if band is None:
                raise ValueError("--band LO HI is required for the bandpass preset")
            edges = [float(band[0]), float(band[1])]
        else:
            if cutoff is None:
                raise ValueError(f"--cutoff is required for the {name} preset")
            edges = float(cutoff)
        pass_zero = name == "lowpass"
        numtaps = int(taps)
        if not pass_zero and numtaps % 2 == 0:
            numtaps += 1  # firwin needs an odd length for a nonzero response at Nyquist
        kernel = firwin(numtaps, edges, fs=fs, window=window, pass_zero=pass_zero)
        if name == "bandpass":
            desc = f"bandpass {edges[0]:g}-{edges[1]:g} Hz"
            params = {"band": edges}
        else:
            desc = f"{name} {edges:g} Hz"
            params = {"cutoff": edges}
        params.update({"fs": fs, "window": window, "design_taps": numtaps})
        return kernel, f"{desc} (fs={fs:g} Hz, {numtaps} taps, {window})", params
    raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")


def from_preset(
    name: str,
    *,
    taps: int | None = None,
    fs: float = 1.0,
    cutoff: float | None = None,
    band=None,
    rho: float = 0.9,
    tau: float = 4.0,
    window: str = "hamming",
    sigma: float | None = 1.0,
    n_taps: int | None = None,
    floor: float = 1e-14,
    nfft: int | None = None,
) -> ProcessSpec:
    """Build a process from one of the named presets (see ``PRESETS``)."""
    name = name.lower()
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    if taps is None:
        taps = _DEFAULT_TAPS[name]
    kernel, label, params = _design_kernel(
        name, taps=taps, fs=fs, cutoff=cutoff, band=band, rho=rho, tau=tau, window=window
    )
    fac = minimum_phase_from_taps(kernel, n_taps=n_taps or kernel.size, nfft=nfft, floor=floor)
    params["preset"] = name
    return _finish(
        fac,
        label,
        fs=fs,
        sigma=sigma,
        design_kernel=kernel,
        target_autocov=autocovariance_from_taps(kernel),
        params=params,
    )
