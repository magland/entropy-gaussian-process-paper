"""Diagnostic figure for a process specification (no estimation performed)."""

from __future__ import annotations

import numpy as np

from .approx import entropy_rate_approx
from .estimate import (
    differential_entropy_bits,
    empirical_symbol_entropy_bits,
    fine_quantization_bits,
    marginal_entropy_bits,
)
from .factor import spectrum_from_autocov
from .pf import log_cell_prob
from .process import generate_quantized
from .spec import ProcessSpec

_EPS = 1e-300


def _db(power):
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=float), _EPS))


def _freq_axis(fs: float):
    return ("frequency (Hz)", fs) if fs != 1.0 else ("frequency (cycles/sample)", 1.0)


def make_figure(
    spec: ProcessSpec,
    delta: float,
    *,
    n: int = 200_000,
    n_show: int = 400,
    seed: int | None = 0,
    nfft: int = 8192,
):
    """Build a six-panel overview of the process and its quantization.

    Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    delta = float(delta)
    rng = np.random.default_rng(seed)
    n = max(n, n_show)
    x, y = generate_quantized(spec.taps, n, delta, rng)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{spec.label}   |   $\\Delta$ = {delta:g} ({delta / spec.sigma:g}$\\sigma$)")

    _panel_timeseries(axes[0, 0], x, y, delta, n_show)
    _panel_histogram(axes[0, 1], y, spec.sigma, delta)
    _panel_spectrum(axes[0, 2], spec, nfft)
    _panel_kernel(axes[1, 0], spec)
    _panel_autocov(axes[1, 1], spec)
    _panel_summary(axes[1, 2], spec, delta, y)

    fig.tight_layout()
    return fig


def _panel_timeseries(ax, x, y, delta, n_show):
    n_show = int(min(n_show, x.size))
    t = np.arange(n_show)
    ax.plot(t, x[:n_show], lw=1.0, color="0.45", label="$X_t$")
    ax.step(t, y[:n_show] * delta, where="mid", lw=1.2, color="C0", label="$\\Delta Y_t$")
    lo = np.floor(x[:n_show].min() / delta - 0.5)
    hi = np.ceil(x[:n_show].max() / delta + 0.5)
    if hi - lo <= 40:  # draw the quantizer cell boundaries when they are legible
        for edge in np.arange(lo, hi + 1) * delta + 0.5 * delta:
            ax.axhline(edge, color="0.85", lw=0.5, zorder=0)
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
    ax.set_title("sample path and quantization")
    ax.legend(loc="upper right", fontsize=8)


def _panel_histogram(ax, y, sigma, delta):
    values, counts = np.unique(y, return_counts=True)
    freq = counts / counts.sum()
    grid = np.arange(values.min(), values.max() + 1)
    lo = (grid - 0.5) * delta / sigma
    exact = np.exp(log_cell_prob(lo, lo + delta / sigma))
    if values.size <= 60:
        ax.bar(values, freq, width=0.85, color="C0", label="empirical")
        ax.plot(grid, exact, "o-", color="C3", ms=3, lw=1, label="Gaussian marginal")
    else:
        ax.step(values, freq, where="mid", color="C0", lw=1.2, label="empirical")
        ax.plot(grid, exact, color="C3", lw=1, label="Gaussian marginal")
    ax.set_xlabel("symbol $y$")
    ax.set_ylabel("probability")
    ax.set_title(f"symbol distribution ({values.size} symbols)")
    ax.legend(fontsize=8)


def _panel_spectrum(ax, spec, nfft):
    label, scale = _freq_axis(spec.fs)
    freq, realized = spec.spectrum(nfft)
    ax.plot(freq, _db(realized), color="C0", lw=1.4, label="minimum-phase model")
    if spec.target_autocov is not None:
        target = np.asarray(spec.target_autocov, dtype=float)
        if target[0] > 0:
            target = target * (spec.sigma**2 / target[0])
            full = spectrum_from_autocov(target, max(nfft, 2 * target.size + 1))
            grid = np.fft.rfftfreq(full.size, d=1.0 / scale)
            ax.plot(
                grid,
                _db(full[: grid.size]),
                color="C3",
                lw=1.0,
                ls="--",
                label="target",
            )
    ax.set_xlabel(label)
    ax.set_ylabel("power (dB)")
    ax.set_title("power spectrum")
    ax.set_ylim(bottom=max(-160.0, _db(np.max(realized)) - 140.0))
    ax.legend(fontsize=8)


def _panel_kernel(ax, spec):
    markers, stems, base = ax.stem(
        np.arange(spec.n_taps), spec.taps, basefmt=" ", linefmt="C0-", markerfmt="C0."
    )
    markers.set_label("minimum-phase")
    if spec.design_kernel is not None:
        design = np.asarray(spec.design_kernel, dtype=float)
        norm = float(np.sqrt(np.dot(design, design)))
        if norm > 0:
            design = design * (spec.sigma / norm)
        ax.plot(np.arange(design.size), design, color="C3", lw=1.0, alpha=0.8, label="as specified")
    ax.legend(fontsize=8)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel("tap $j$")
    ax.set_ylabel("$h_j$")
    ax.set_title(f"convolution kernel ($L$ = {spec.n_taps}, $h_0$ = {spec.h0:.4g})")


def _panel_autocov(ax, spec):
    max_lag = min(spec.n_taps - 1, 200)
    realized = spec.autocovariance(max_lag=max_lag)
    lags = np.arange(realized.size)
    ax.plot(lags, realized, color="C0", lw=1.4, label="minimum-phase model")
    if spec.target_autocov is not None:
        target = np.asarray(spec.target_autocov, dtype=float)
        if target[0] > 0:
            target = target * (spec.sigma**2 / target[0])
        keep = min(target.size, realized.size)
        ax.plot(
            np.arange(keep), target[:keep], color="C3", ls="--", lw=1.0, label="target"
        )
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel("lag $k$")
    ax.set_ylabel("$\\gamma_k$")
    ax.set_title("autocovariance")
    ax.legend(fontsize=8)


def _panel_summary(ax, spec, delta, y):
    ax.axis("off")
    upper = marginal_entropy_bits(spec.sigma, delta)
    lower = max(0.0, fine_quantization_bits(spec.h0, delta))
    lines = [
        f"L (taps)          {spec.n_taps}",
        f"sigma             {spec.sigma:.6g}",
        f"h0                {spec.h0:.6g}",
        f"delta             {delta:.6g}",
        f"delta / sigma     {delta / spec.sigma:.6g}",
        f"delta / h0        {delta / spec.h0:.6g}",
        "",
        f"autocov error     {spec.autocov_error:.3e}",
        f"floored bins      {100 * spec.floored_fraction:.2f} %",
        "",
        f"diff entropy rate {differential_entropy_bits(spec.h0):.4f} bits  [hbar]",
        f"symbols observed  {np.unique(y).size}",
        f"order-0 entropy   {empirical_symbol_entropy_bits(y):.4f} bits",
        "",
        "entropy rate (bits/sample)",
        f"  lower  {lower:.4f}   [hbar - log2 delta]",
        f"  approx {entropy_rate_approx(spec.taps, delta):.4f}   [S + delta^2/12]",
        f"  upper  {upper:.4f}   [H(Y_1)]",
    ]
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=ax.transAxes,
    )
    ax.set_title("summary")
