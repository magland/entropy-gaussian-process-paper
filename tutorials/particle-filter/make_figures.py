#!/usr/bin/env python
"""Figures for the particle filter tutorial (README.md in this directory).

Everything here is self-contained apart from the process specification and the
analytical approximation, which come from the ``egp`` package.  The filters are
written out in full rather than imported, since the point of the tutorial is to
show what they do; the production versions live in ``egp/src/egp/pf.py``.

Run with::

    python make_figures.py            # all figures
    python make_figures.py 3 9        # only fig03 and fig09
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import egp
from egp.pf import log_cell_prob, sample_truncnorm

FIGURES = Path(__file__).resolve().parent / "figures"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2261b8"
LIGHTBLUE = "#8fb9f0"
ORANGE = "#c26d24"
PURPLE = "#7b4fa6"
GREEN = "#3d7a4a"
RED = "#b83232"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")


def normal_pdf(x, mean=0.0, sd=1.0):
    z = (np.asarray(x) - mean) / sd
    return np.exp(-0.5 * z * z) / (sd * np.sqrt(2 * np.pi))


def systematic_resample(weights, rng):
    """Ancestor indices drawn with probability proportional to ``weights``."""
    n = weights.size
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    u = (rng.random() + np.arange(n)) / n
    return np.clip(np.searchsorted(cumulative, u), 0, n - 1)


# ----------------------------------------------------------------------------
# Part I: a textbook scalar state-space model, used for the refresher figures.
#
#   X_t = A X_{t-1} + SW eps_t,      Y_t = X_t + SV eta_t
#
# Linear and Gaussian, so the Kalman filter gives the exact posterior and the
# exact likelihood to compare the particle filter against.
# ----------------------------------------------------------------------------

A, SW, SV = 0.9, 0.5, 0.7
S0 = SW / np.sqrt(1 - A**2)  # stationary standard deviation of X


def toy_simulate(n, rng):
    x = np.empty(n)
    xt = rng.normal(0.0, S0)
    for t in range(n):
        xt = A * xt + SW * rng.standard_normal()
        x[t] = xt
    y = x + SV * rng.standard_normal(n)
    return x, y


def kalman(y):
    """Exact filtering means/variances and log-likelihood (nats)."""
    n = len(y)
    m_pred, p_pred = 0.0, S0**2
    pred_mean, pred_var = np.empty(n), np.empty(n)
    post_mean, post_var = np.empty(n), np.empty(n)
    loglik = 0.0
    for t in range(n):
        pred_mean[t], pred_var[t] = m_pred, p_pred
        s = p_pred + SV**2
        loglik += -0.5 * (np.log(2 * np.pi * s) + (y[t] - m_pred) ** 2 / s)
        k = p_pred / s
        m = m_pred + k * (y[t] - m_pred)
        p = (1 - k) * p_pred
        post_mean[t], post_var[t] = m, p
        m_pred, p_pred = A * m, A * A * p + SW**2
    return dict(
        pred_mean=pred_mean,
        pred_var=pred_var,
        post_mean=post_mean,
        post_var=post_var,
        loglik=loglik,
    )


def bootstrap_pf(y, n_particles, rng, resample=True, snapshot_at=None, keep_cloud=False):
    """The textbook bootstrap filter: propagate blindly, weight, resample.

    Returns the running log-likelihood, the per-step effective sample size, and
    (optionally) the particle cloud and a snapshot of one step's three stages.
    """
    n = len(y)
    x = rng.normal(0.0, S0, n_particles)  # prior for X_1
    logw = np.zeros(n_particles)  # only accumulates when resample=False
    loglik = 0.0
    ess = np.empty(n)
    mean = np.empty(n)
    cloud = np.empty((n, n_particles)) if keep_cloud else None
    snap = {}

    for t in range(n):
        step_logw = -0.5 * ((y[t] - x) / SV) ** 2  # log N(y_t; x_t, SV^2), up to a constant
        logw = logw + step_logw
        w = np.exp(logw - logw.max())
        w /= w.sum()
        loglik += np.log(np.mean(np.exp(step_logw - step_logw.max()))) + step_logw.max()
        loglik += -0.5 * np.log(2 * np.pi * SV**2)
        ess[t] = 1.0 / np.sum(w**2) / n_particles
        mean[t] = np.dot(w, x)
        if keep_cloud:
            cloud[t] = x

        if t == snapshot_at:
            snap = dict(prior=x.copy(), weights=w.copy())

        if resample:
            a = systematic_resample(w, rng)
            x_post = x[a]
            logw = np.zeros(n_particles)
        else:
            x_post = x
        if t == snapshot_at:
            snap["posterior"] = x_post.copy()
        x = A * x_post + SW * rng.standard_normal(n_particles)
        if t == snapshot_at:
            snap["next_prior"] = x.copy()

    return dict(loglik=loglik, ess=ess, mean=mean, cloud=cloud, snapshot=snap)


# ----------------------------------------------------------------------------
# Part II: the paper's model.
#
#   X_t = sum_j h_j W_{t-j},   W iid N(0,1),   Y_t = round(X_t / delta)
#
# State: the last L-1 innovations.  Two filters for it, the fully adapted one
# of the paper and a bootstrap filter for comparison.
# ----------------------------------------------------------------------------


def adapted_filter(y, taps, delta, n_particles, rng, snapshot_at=None):
    """The fully adapted filter of Section 4.3, instrumented for plotting.

    Each particle is one guess at the last L-1 innovations, so the whole cloud
    is the ``(n_particles, L-1)`` array ``u``.  With ``snapshot_at`` set, the
    state of that array at the given step, before resampling, is returned too.
    """
    h0, tail = float(taps[0]), np.asarray(taps[1:], dtype=float)
    n, width = len(y), delta / h0
    u = rng.standard_normal((n_particles, tail.size))  # u[:, k] holds W_{t-1-k}
    step_loglik, ess = np.empty(n), np.empty(n)
    snap = {}

    for t in range(n):
        mu = u @ tail  # prediction of X_t from each particle's past
        lo = ((y[t] - 0.5) * delta - mu) / h0
        hi = lo + width
        log_alpha = log_cell_prob(lo, hi)  # log P(Y_t = y_t | particle)

        peak = log_alpha.max()
        w = np.exp(log_alpha - peak)
        step_loglik[t] = peak + np.log(w.sum()) - np.log(n_particles)
        ess[t] = w.sum() ** 2 / np.dot(w, w) / n_particles
        if t == snapshot_at:
            snap = dict(u=u.copy(), mu=mu.copy(), alpha=np.exp(log_alpha))

        a = systematic_resample(w, rng)
        w_new = sample_truncnorm(lo[a], hi[a], rng)  # the exact conditional for W_t
        u = np.concatenate([w_new[:, None], u[a][:, :-1]], axis=1)

    return dict(step_loglik=step_loglik, ess=ess, loglik=float(step_loglik.sum()), snapshot=snap)


def bootstrap_filter_quantized(y, taps, delta, n_particles, rng):
    """The same model filtered blindly: propose W_t ~ N(0,1), then check the cell.

    The weight is an indicator, so the effective sample size is exactly the
    fraction of particles that landed in the observed cell.
    """
    h0, tail = float(taps[0]), np.asarray(taps[1:], dtype=float)
    n = len(y)
    u = rng.standard_normal((n_particles, tail.size))
    step_loglik, ess = np.full(n, np.nan), np.full(n, np.nan)
    died_at = -1

    for t in range(n):
        mu = u @ tail
        w_prop = rng.standard_normal(n_particles)
        x = mu + h0 * w_prop
        inside = ((y[t] - 0.5) * delta <= x) & (x < (y[t] + 0.5) * delta)
        n_in = int(inside.sum())
        if n_in == 0:  # every particle contradicts the observation
            died_at = t
            break
        step_loglik[t] = np.log(n_in / n_particles)
        ess[t] = n_in / n_particles

        a = rng.choice(np.flatnonzero(inside), size=n_particles, replace=True)
        u = np.concatenate([w_prop[a][:, None], u[a][:, :-1]], axis=1)

    return dict(step_loglik=step_loglik, ess=ess, died_at=died_at)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------


def fig01_setup():
    """The problem: a smooth Gaussian process seen through a uniform quantizer."""
    spec = egp.from_preset("gaussian", tau=1.0, taps=9)
    delta, n = 0.5, 70
    rng = np.random.default_rng(3)
    x, y = egp.generate_quantized(spec.taps, n, delta, rng)
    t = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True, height_ratios=[2, 1])
    ax = axes[0]
    for k in range(int(np.floor(x.min() / delta)) - 1, int(np.ceil(x.max() / delta)) + 2):
        ax.axhline((k + 0.5) * delta, color=GRID, linewidth=0.7, zorder=0)
    ax.step(t, y * delta, where="mid", color=ORANGE, linewidth=1.4, label=r"$\Delta y_t$ (what is stored)")
    ax.plot(t, x, color=BLUE, linewidth=1.6, label=r"$X_t$ (the process)")
    ax.plot(t, x, "o", color=BLUE, markersize=2.5)
    ax.set_ylabel(r"amplitude / $\sigma$")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title(
        r"(a) Gaussian-smoothed noise ($\tau = 1$, $\sigma = 1$) and its quantization at "
        r"$\Delta = 0.5\sigma$; gray lines are the cell boundaries",
        loc="left",
    )

    ax = axes[1]
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.vlines(t, 0, y, color=MUTED, linewidth=0.9)
    ax.plot(t, y, "o", color=ORANGE, markersize=3.5)
    ax.set_ylabel(r"symbol $y_t$")
    ax.set_xlabel("time step $t$")
    ax.set_title(r"(b) the integer sequence whose entropy rate we want", loc="left")
    fig.tight_layout()
    save(fig, "fig01_setup")


def fig02_importance():
    """Monte Carlo refresher: importance sampling and resampling."""
    obs, sd = 1.2, 0.5
    post_var = 1.0 / (1.0 + 1.0 / sd**2)
    post_mean = post_var * obs / sd**2
    grid = np.linspace(-3.2, 3.2, 400)
    rng = np.random.default_rng(1)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    ax.plot(grid, normal_pdf(grid), color=MUTED, linewidth=1.4, label=r"prior $p(x)$")
    like = normal_pdf(obs, grid, sd)
    ax.plot(grid, like / like.max() * 0.42, color=ORANGE, linewidth=1.4, label=r"likelihood $p(y\mid x)$")
    ax.plot(grid, normal_pdf(grid, post_mean, np.sqrt(post_var)), color=BLUE, linewidth=1.4,
            label=r"posterior $p(x\mid y)$")
    n_show = 200
    xs = rng.standard_normal(n_show)
    w = normal_pdf(obs, xs, sd)
    ax.scatter(xs, np.full_like(xs, -0.06), s=3 + 90 * w / w.max(), facecolor="none",
               edgecolor=INK, linewidth=0.6, alpha=0.6)
    ax.axhline(-0.06, color=AXIS, linewidth=0.6)
    resampled = xs[systematic_resample(w, rng)]
    ax.scatter(resampled, np.full_like(resampled, -0.15), s=7, facecolor="none",
               edgecolor=PURPLE, linewidth=0.6, alpha=0.6)
    ax.axhline(-0.15, color=AXIS, linewidth=0.6)
    ax.text(-3.15, -0.045, "weighted", fontsize=7.5, color=INK2, va="bottom")
    ax.text(-3.15, -0.135, "resampled", fontsize=7.5, color=PURPLE, va="bottom")
    ax.set_ylim(-0.21, 0.95)
    ax.set_xlabel("$x$")
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"(a) {n_show} draws from the prior, sized by weight", loc="left")

    ax = axes[1]
    big = rng.standard_normal(4000)
    wb = normal_pdf(obs, big, sd)
    wb /= wb.sum()
    ax.hist(big, bins=45, weights=wb, density=True, color=LIGHTBLUE, alpha=0.85,
            label="weighted particles")
    idx = systematic_resample(wb, rng)
    ax.hist(big[idx], bins=45, density=True, histtype="step", color=PURPLE, linewidth=1.2,
            label="after resampling")
    ax.plot(grid, normal_pdf(grid, post_mean, np.sqrt(post_var)), color=BLUE, linewidth=1.6,
            label="exact posterior")
    ax.set_xlim(-1.2, 3.0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=8)
    ess = 1.0 / np.sum(wb**2)
    ax.set_title(f"(b) 4000 draws; ESS = {ess:.0f}", loc="left")
    fig.tight_layout()
    save(fig, "fig02_importance")


def fig03_one_step():
    """One step of the bootstrap filter, stage by stage."""
    rng = np.random.default_rng(7)
    n, t0, n_particles = 12, 8, 600
    x_true, y = toy_simulate(n, rng)
    kf = kalman(y)
    res = bootstrap_pf(y, n_particles, np.random.default_rng(11), snapshot_at=t0)
    snap = res["snapshot"]
    grid = np.linspace(-3.5, 3.5, 400)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)

    ax = axes[0]
    ax.hist(snap["prior"], bins=40, density=True, color=LIGHTBLUE, alpha=0.9)
    ax.plot(grid, normal_pdf(grid, kf["pred_mean"][t0], np.sqrt(kf["pred_var"][t0])), color=BLUE, linewidth=1.5)
    ax.set_title(
        r"(a) predict: propagate every particle through $X_t = 0.9X_{t-1} + 0.5\varepsilon_t$"
        "\n"
        r"     the cloud now approximates $p(x_t \mid y_{1:t-1})$ (blue curve: exact)",
        loc="left",
    )
    ax.set_yticks([])

    ax = axes[1]
    like = normal_pdf(y[t0], grid, SV)
    ax.plot(grid, like / like.max(), color=ORANGE, linewidth=1.5, label=r"$p(y_t \mid x_t)$")
    ax.axvline(y[t0], color=ORANGE, linestyle=":", linewidth=1.2, label=r"observation $y_t$")
    show = np.random.default_rng(0).choice(n_particles, 120, replace=False)
    ax.scatter(snap["prior"][show], np.full(show.size, -0.14),
               s=4 + 160 * snap["weights"][show] / snap["weights"].max(),
               facecolor="none", edgecolor=INK, linewidth=0.7, alpha=0.75)
    ax.axhline(-0.14, color=AXIS, linewidth=0.6)
    ax.set_ylim(-0.30, 1.15)
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        r"(b) weight: each particle gets $w^{(i)} \propto p(y_t \mid x_t^{(i)})$"
        "\n"
        r"     particles that predicted the observation well get more weight",
        loc="left",
    )

    ax = axes[2]
    ax.hist(snap["posterior"], bins=40, density=True, color=LIGHTBLUE, alpha=0.9)
    ax.plot(grid, normal_pdf(grid, kf["post_mean"][t0], np.sqrt(kf["post_var"][t0])), color=BLUE, linewidth=1.5)
    ax.axvline(x_true[t0], color=GREEN, linewidth=1.2, linestyle="--", label="true state")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_xlim(-3.0, 1.6)
    ax.set_title(
        r"(c) resample: draw $N$ particles with probability $\propto w^{(i)}$"
        "\n"
        r"     equal weights again, and the cloud approximates $p(x_t \mid y_{1:t})$",
        loc="left",
    )
    fig.tight_layout()
    save(fig, "fig03_one_step")


def fig04_tracking():
    """The filter running: the cloud follows the hidden state."""
    rng = np.random.default_rng(4)
    n, n_particles = 60, 400
    x_true, y = toy_simulate(n, rng)
    kf = kalman(y)
    res = bootstrap_pf(y, n_particles, np.random.default_rng(5), keep_cloud=True)
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.scatter(np.repeat(t, n_particles), res["cloud"].ravel(), s=3, color=BLUE, alpha=0.06,
               linewidths=0, label="particles")
    ax.plot(t, y, "o", color=ORANGE, markersize=3, label="observations $y_t$")
    ax.plot(t, x_true, color=GREEN, linewidth=1.8, label="true state $x_t$")
    ax.plot(t, res["mean"], color=INK, linewidth=1.2, label="particle mean")
    ax.plot(t, kf["post_mean"], color=RED, linewidth=1.0, linestyle="--", label="Kalman mean (exact)")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("$x$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5, fontsize=8)
    ax.set_title(f"Bootstrap filter with $N = {n_particles}$ particles", loc="left")
    fig.tight_layout()
    save(fig, "fig04_tracking")


def fig05_degeneracy():
    """Why resampling: without it the weights concentrate on one particle."""
    rng = np.random.default_rng(12)
    n, n_particles = 60, 500
    _, y = toy_simulate(n, rng)
    with_r = bootstrap_pf(y, n_particles, np.random.default_rng(13), resample=True)
    without_r = bootstrap_pf(y, n_particles, np.random.default_rng(13), resample=False)
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.semilogy(t, with_r["ess"], color=BLUE, linewidth=1.6, label="resampling at every step")
    ax.semilogy(t, without_r["ess"], color=ORANGE, linewidth=1.6, label="no resampling")
    ax.axhline(1.0 / n_particles, color=MUTED, linewidth=0.9, linestyle=":")
    ax.text(1.0, 1.25 / n_particles, "one surviving particle", ha="left", va="bottom",
            fontsize=8, color=MUTED)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("ESS / $N$")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(axis="y")
    ax.set_title(f"Weight degeneracy, $N = {n_particles}$", loc="left")
    fig.tight_layout()
    save(fig, "fig05_degeneracy")


def fig06_jensen():
    """The likelihood is unbiased; its logarithm is not."""
    rng = np.random.default_rng(21)
    n, n_reps = 300, 300
    _, y = toy_simulate(n, rng)
    exact = kalman(y)["loglik"]

    counts = [8, 16, 32, 64, 128, 256, 512, 1024]
    bias, err, samples = [], [], {}
    for n_particles in counts:
        vals = np.array([
            bootstrap_pf(y, n_particles, np.random.default_rng(1000 + n_particles * 97 + r))["loglik"]
            for r in range(n_reps)
        ])
        samples[n_particles] = vals
        d = (exact - vals) / (n * np.log(2))  # bits per sample, positive = overestimate of H
        bias.append(d.mean())
        err.append(d.std(ddof=1) / np.sqrt(n_reps))
        print(f"    N = {n_particles:5d}   bias = {d.mean() * 1000:7.2f} mbits/sample")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for n_particles, color in ((16, ORANGE), (64, PURPLE), (512, BLUE)):
        ax.hist((exact - samples[n_particles]) / (n * np.log(2)) * 1000, bins=35, density=True,
                histtype="stepfilled", alpha=0.45, color=color, label=f"$N = {n_particles}$")
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.text(0.02, 0.94, "exact value", transform=ax.transAxes, fontsize=8, color=INK)
    ax.set_xlabel("overestimate of the rate (mbits/sample)")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"(a) {n_reps} runs on one fixed sequence, $n = {n}$", loc="left")

    ax = axes[1]
    ax.errorbar(counts, np.array(bias) * 1000, yerr=np.array(err) * 1000, marker="o",
                markersize=4, color=BLUE, linewidth=1.6, capsize=2.5)
    guide = np.array(counts, dtype=float)
    ax.plot(guide, bias[0] * 1000 * counts[0] / guide, color=MUTED, linestyle="--", linewidth=1.0,
            label=r"$\propto 1/N$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("particles $N$")
    ax.set_ylabel("bias (mbits/sample)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid()
    ax.set_title("(b) the bias decays as $1/N$", loc="left")
    fig.tight_layout()
    save(fig, "fig06_jensen")


def fig07_model():
    """The moving-average state space and the interval-valued observation."""
    spec = egp.from_preset("gaussian", tau=1.0, taps=9)
    taps, h0 = spec.taps, spec.h0
    n_taps = taps.size
    delta = 0.5
    rng = np.random.default_rng(2)
    w = rng.standard_normal(18)
    t0 = 14
    window = w[t0 - n_taps + 1 : t0 + 1][::-1]  # W_t, W_{t-1}, ..., W_{t-L+1}
    xt = float(np.dot(taps, window))
    mu = float(np.dot(taps[1:], window[1:]))

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.75, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.vlines(np.arange(n_taps), 0, taps, color=MUTED, linewidth=1.0)
    ax.plot(np.arange(n_taps), taps, "o", color=BLUE, markersize=4)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.annotate(f"$h_0 = {h0:.3f}$", xy=(0, taps[0]), xytext=(1.4, taps[0] * 0.95),
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, linewidth=0.8))
    ax.set_xlabel("$j$")
    ax.set_ylabel("$h_j$")
    ax.set_title(r"(a) minimum-phase taps, $L = 9$", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    grid = np.linspace(mu - 4 * h0, mu + 4 * h0, 400)
    ax.plot(grid, normal_pdf(grid, mu, h0), color=BLUE, linewidth=1.5)
    y_obs = int(np.floor(xt / delta + 0.5))
    lo, hi = (y_obs - 0.5) * delta, (y_obs + 0.5) * delta
    band = (grid >= lo) & (grid <= hi)
    ax.fill_between(grid[band], 0, normal_pdf(grid[band], mu, h0), color=LIGHTBLUE, alpha=0.9)
    for edge, name in ((lo, r"$(y_t-\frac{1}{2})\Delta$"), (hi, r"$(y_t+\frac{1}{2})\Delta$")):
        ax.axvline(edge, color=ORANGE, linewidth=1.1)
        ax.text(edge, ax.get_ylim()[1] * 0.02, name, rotation=90, fontsize=7.5, color=ORANGE,
                ha="right", va="bottom")
    alpha = float(np.exp(log_cell_prob((lo - mu) / h0, (hi - mu) / h0)))
    ax.text(mu, normal_pdf(mu, mu, h0) * 0.45, rf"$\alpha = {alpha:.2f}$", ha="center", fontsize=9, color=INK)
    ax.set_xlabel("$x$")
    ax.set_yticks([])
    ax.set_title(r"(b) $X_t \mid \mathrm{state} \sim \mathcal{N}(\mu, h_0^2)$; the cell has mass $\alpha$",
                 loc="left")

    ax = fig.add_subplot(gs[1, :])
    idx = np.arange(w.size)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.axvspan(t0 - n_taps + 0.5, t0 - 0.5, color="#f2f4f8", zorder=0)
    ax.axvspan(t0 - 0.5, t0 + 0.5, color="#fdf0e3", zorder=0)
    ax.vlines(idx, 0, w, color=MUTED, linewidth=1.0)
    ax.plot(idx, w, "o", color=MUTED, markersize=3.5)
    inwin = idx[(idx >= t0 - n_taps + 1) & (idx <= t0 - 1)]
    ax.plot(inwin, w[inwin], "o", color=BLUE, markersize=4.5)
    ax.plot([t0], [w[t0]], "o", color=ORANGE, markersize=5.5)
    ax.set_xticks(idx[::2])
    ax.set_xlabel("innovation index")
    ax.set_ylabel(r"$W$")
    ax.text((t0 - n_taps + 1 + t0 - 1) / 2, ax.get_ylim()[1] * 0.82,
            r"the state: $(W_{t-L+1}, \dots, W_{t-1})$, $L-1 = 8$ numbers",
            ha="center", fontsize=8.5, color=BLUE)
    ax.text(t0 + 0.7, ax.get_ylim()[0] * 0.85, r"new innovation $W_t$", ha="left", fontsize=8.5, color=ORANGE)
    ax.set_title(
        r"(c) $X_t = \sum_{j=0}^{L-1} h_j W_{t-j}$: a sliding window of innovations. "
        r"Everything the past tells us about $X_t$ is in the shaded state.",
        loc="left",
    )
    save(fig, "fig07_model")


def fig08_particles():
    """What a particle is here: one guess at the recent innovations."""
    spec = egp.from_preset("gaussian", tau=1.0, taps=9)
    taps, h0, n_taps = spec.taps, spec.h0, spec.n_taps
    delta, n, n_particles, t0 = 0.5, 60, 2000, 40

    rng = np.random.default_rng(12)
    w_true = rng.standard_normal(n + n_taps - 1)
    x = np.convolve(w_true, taps, mode="valid")[:n]  # x[k] uses w_true[k+L-1-j] as W_{t-j}
    y = np.floor(x / delta + 0.5).astype(np.int64)
    true_mu = float(np.dot(taps[1:], w_true[t0 + n_taps - 2 :: -1][: n_taps - 1]))

    snap = adapted_filter(y, taps, delta, n_particles, np.random.default_rng(4),
                          snapshot_at=t0)["snapshot"]
    alpha, mu, u = snap["alpha"], snap["mu"], snap["u"]

    order = np.argsort(alpha)
    picks = [order[int(q * (n_particles - 1))] for q in (0.999, 0.85, 0.5, 0.2, 0.03)]
    colors = ["#0d3a75", "#2261b8", "#4f95e6", "#8fb9f0", "#b9b8b1"]
    lo_x, hi_x = (y[t0] - 0.5) * delta, (y[t0] + 0.5) * delta

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), width_ratios=[1.1, 1])

    ax = axes[0]
    lags = np.arange(1, n_taps)
    scale = 0.3
    for row, (i, color) in enumerate(zip(picks, colors)):
        base = -row
        ax.plot([lags[0] - 0.4, lags[-1] + 0.4], [base, base], color=AXIS, linewidth=0.6, zorder=0)
        ax.vlines(lags, base, base + scale * u[i], color=color, linewidth=1.2)
        ax.plot(lags, base + scale * u[i], "o", color=color, markersize=3.5)
    ax.set_yticks([-row for row in range(len(picks))])
    ax.set_yticklabels([rf"$\alpha = {alpha[i]:.3f}$" for i in picks])
    for label, color in zip(ax.get_yticklabels(), colors):
        label.set_color(color)
    ax.set_xticks(lags)
    ax.set_xlabel(r"lag $j$: the particle's guess at $W_{t-j}$")
    ax.set_title(
        r"(a) five particles from a cloud of 2000; each one is"
        "\n"
        r"     a guess at the last $L-1 = 8$ innovations.",
        loc="left",
    )

    ax = axes[1]
    cell = ax.axvspan(lo_x, hi_x, color="#fdf0e3", zorder=0, label="observed cell")
    grid = np.linspace(min(mu[picks].min(), lo_x) - 3.2 * h0,
                       max(mu[picks].max(), hi_x) + 3.2 * h0, 600)
    for i, color in zip(picks, colors):
        ax.plot(grid, normal_pdf(grid, mu[i], h0), color=color, linewidth=1.5)
    show = np.random.default_rng(0).choice(n_particles, 400, replace=False)
    rug, = ax.plot(mu[show], np.full(400, -0.06), "|", color=MUTED, markersize=5, alpha=0.5,
                   label=r"all 2000 predictions $\mu^{(i)}$")
    truth = ax.axvline(true_mu, color=INK, linewidth=1.0, linestyle="--",
                       label="the true prediction")
    ax.legend(handles=[cell, truth, rug], loc="upper right", fontsize=7.5)
    ax.set_ylim(-0.12, 1.45 * normal_pdf(0.0, 0.0, h0))
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_title(
        r"(b) each guess predicts a different $X_t \sim \mathcal{N}(\mu^{(i)}, h_0^2)$;"
        "\n"
        r"     the weight $\alpha^{(i)}$ is the mass it puts on the cell.",
        loc="left",
    )
    fig.tight_layout()
    save(fig, "fig08_particles")


def fig09_adapted():
    """Blind proposal versus the fully adapted one."""
    spec = egp.from_preset("gaussian", tau=1.0, taps=9)
    taps, h0 = spec.taps, spec.h0
    rng = np.random.default_rng(9)

    fig = plt.figure(figsize=(7.4, 3.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.7, wspace=0.52)

    # (a), (b): one particle, one step, at delta / h0 = 0.4
    mu, delta = 0.0, 0.4 * h0
    lo, hi = -0.1 * delta, 0.9 * delta  # an off-center cell
    grid = np.linspace(mu - 3.2 * h0, mu + 3.2 * h0, 800)
    dens = normal_pdf(grid, mu, h0)
    band = (grid >= lo) & (grid <= hi)
    mass = float(np.exp(log_cell_prob((lo - mu) / h0, (hi - mu) / h0)))
    draws = rng.normal(mu, h0, 200)
    inside = (draws >= lo) & (draws < hi)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(grid, dens, color=BLUE, linewidth=1.5)
    ax.fill_between(grid[band], 0, dens[band], color=LIGHTBLUE, alpha=0.9)
    ax.plot(draws[~inside], np.full((~inside).sum(), -0.11 * dens.max()), "|", color=MUTED, markersize=6)
    ax.plot(draws[inside], np.full(inside.sum(), -0.11 * dens.max()), "|", color=ORANGE, markersize=6)
    ax.set_ylim(-0.2 * dens.max(), dens.max() * 1.1)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_yticks([])
    ax.set_title(rf"(a) blind proposal: {inside.sum()}/200 draws in the cell", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(grid[band], dens[band] / mass, color=ORANGE, linewidth=1.8)
    ax.fill_between(grid[band], 0, dens[band] / mass, color=ORANGE, alpha=0.25)
    adapted = sample_truncnorm(np.full(200, (lo - mu) / h0), np.full(200, (hi - mu) / h0), rng) * h0 + mu
    top = (dens[band] / mass).max()
    ax.plot(adapted, np.full(200, -0.11 * top), "|", color=ORANGE, markersize=6)
    ax.set_ylim(-0.2 * top, top * 1.1)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_title(rf"(b) adapted proposal: 200/200 draws", loc="left")

    # (c) effective sample size against delta / h0, both filters
    ax = fig.add_subplot(gs[:, 1])
    ratios = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    n, n_particles = 200, 1000
    ess_ad, ess_bs, died = [], [], []
    for ratio in ratios:
        delta = ratio * h0
        gen = np.random.default_rng(100 + int(ratio * 8))
        _, y = egp.generate_quantized(taps, n, delta, gen)
        a = adapted_filter(y, taps, delta, n_particles, np.random.default_rng(31))
        b = bootstrap_filter_quantized(y, taps, delta, n_particles, np.random.default_rng(31))
        ess_ad.append(np.mean(a["ess"]))
        ess_bs.append(np.nanmean(b["ess"]) if b["died_at"] != 0 else np.nan)
        died.append(b["died_at"])
        print(f"    delta/h0 = {ratio:5.2f}   adapted ESS {ess_ad[-1]:.3f}   "
              f"bootstrap ESS {ess_bs[-1]:.4f}   died at {b['died_at']}")
    ax.loglog(ratios, ess_ad, "o-", color=BLUE, markersize=4.5, linewidth=1.6, label="fully adapted")
    ax.loglog(ratios, ess_bs, "s-", color=ORANGE, markersize=4.5, linewidth=1.6, label="bootstrap")
    for ratio, e, d in zip(ratios, ess_bs, died):
        if d >= 0:
            ax.plot([ratio], [e], "x", color=RED, markersize=9, markeredgewidth=1.6)
    ax.plot([], [], "x", color=RED, markersize=8, linestyle="none", label="bootstrap filter died")
    ax.set_xlabel(r"$\Delta / h_0$")
    ax.set_ylabel("mean ESS / $N$")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(which="both")
    ax.set_title(f"(c) mean ESS ($N = {n_particles}$, $n = {n}$)", loc="left")
    save(fig, "fig09_adapted")


def fig10_collapse():
    """Lost lock: the failure the effective sample size does not see."""
    spec = egp.from_preset("gaussian", tau=1.5, taps=13)
    taps, h0 = spec.taps, spec.h0
    delta, n = 0.5, 8000
    rng = np.random.default_rng(5)
    _, y = egp.generate_quantized(taps, n, delta, rng)
    limit = float(np.log(min(1.0, delta / h0)) - 25.0)

    runs = {}
    for n_particles, color, label in ((300, ORANGE, "$N = 300$"), (3000, BLUE, "$N = 3000$")):
        res = adapted_filter(y, taps, delta, n_particles, np.random.default_rng(23))
        runs[n_particles] = (res, color, label)
        rate = -res["loglik"] / (n * np.log(2))
        lost = float(np.mean(res["step_loglik"] < limit))
        print(f"    N = {n_particles:5d}   estimate {rate:12.4f} bits/sample   "
              f"lost-lock steps {lost:6.1%}   mean ESS {np.mean(res['ess']):.2f}")

    t = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    ax = axes[0]
    for n_particles, (res, color, label) in runs.items():
        ax.plot(t, res["step_loglik"], color=color, linewidth=0.5, alpha=0.8, label=label)
    ax.axhline(limit, color=RED, linewidth=1.2, linestyle="--", label="lost-lock threshold")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("step log-likelihood (nats)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.set_title(
        r"(a) $\log P(y_t \mid y_{1:t-1})$ per step, Gaussian $\tau = 1.5$ at $\Delta = 0.5\sigma$",
        loc="left",
    )

    ax = axes[1]
    window = 200
    kernel = np.ones(window) / window
    for n_particles, (res, color, label) in runs.items():
        ax.plot(t, res["ess"], color=color, linewidth=0.4, alpha=0.15)
        smooth = np.convolve(res["ess"], kernel, mode="valid")
        ax.plot(t[window - 1:], smooth, color=color, linewidth=1.8,
                label=f"{label}, mean {np.mean(res['ess']):.0%}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("ESS / $N$")
    ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9)
    ax.set_title("(b) the effective sample size gives no warning (thick lines: 200-step average)",
                 loc="left")
    fig.tight_layout()
    save(fig, "fig10_collapse")


def fig11_estimate():
    """From per-step log-likelihoods to the entropy rate."""
    spec = egp.from_preset("ma", taps=8)
    taps, delta = spec.taps, 0.5
    approx = egp.entropy_rate_approx(taps, delta)
    n = 8000

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    first = 10  # the first few steps are far off scale and say nothing
    steps = np.arange(1, n + 1)
    curves = []
    for rep in range(3):
        gen = np.random.default_rng(50 + rep)
        _, y = egp.generate_quantized(taps, n, delta, gen)
        res = adapted_filter(y, taps, delta, 1000, np.random.default_rng(60 + rep))
        running = -np.cumsum(res["step_loglik"]) / (steps * np.log(2))
        curves.append(running)
        ax.plot(steps[first - 1:], running[first - 1:], linewidth=1.2, alpha=0.9,
                color=[BLUE, ORANGE, PURPLE][rep])
    ax.axhline(approx, color=INK, linewidth=1.0, linestyle="--", label="analytical approximation")
    shown = np.concatenate([c[first - 1:] for c in curves])
    span = shown.max() - shown.min()
    ax.set_ylim(shown.min() - 0.08 * span, shown.max() + 0.28 * span)
    ax.set_xlim(first, n)
    ax.set_xscale("log")
    ax.set_xlabel("steps $t$ averaged")
    ax.set_ylabel("bits/sample")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        r"(a) $-\frac{1}{t}\log_2 P(y_{1:t})$ on three sequences."
        "\n"
        r"     MA(8) at $\Delta = 0.5\sigma$, $N = 1000$.",
        loc="left",
    )

    ax = axes[1]
    counts = [10, 30, 100, 300, 1000, 3000]
    means, errs = [], []
    for n_particles in counts:
        vals = []
        for rep in range(4):
            gen = np.random.default_rng(70 + rep)
            _, y = egp.generate_quantized(taps, 4000, delta, gen)
            res = adapted_filter(y, taps, delta, n_particles, np.random.default_rng(80 + rep))
            vals.append(-res["loglik"] / (4000 * np.log(2)))
        means.append(np.mean(vals))
        errs.append(np.std(vals, ddof=1) / 2.0)
        print(f"    N = {n_particles:5d}   estimate {means[-1]:.4f} +/- {errs[-1]:.4f} bits/sample")
    ax.errorbar(counts, means, yerr=errs, marker="o", markersize=4, color=BLUE, linewidth=1.6, capsize=2.5)
    ax.axhline(approx, color=INK, linewidth=1.0, linestyle="--", label="analytical approximation")
    ax.set_xscale("log")
    ax.set_xlabel("particles $N$")
    ax.set_ylabel("bits/sample")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y")
    ax.set_title("(b) the estimate falls toward the rate as $N$ grows", loc="left")
    fig.tight_layout()
    save(fig, "fig11_estimate")


FIGURES_ALL = {
    1: fig01_setup,
    2: fig02_importance,
    3: fig03_one_step,
    4: fig04_tracking,
    5: fig05_degeneracy,
    6: fig06_jensen,
    7: fig07_model,
    8: fig08_particles,
    9: fig09_adapted,
    10: fig10_collapse,
    11: fig11_estimate,
}


def main(argv):
    wanted = [int(a) for a in argv[1:]] or sorted(FIGURES_ALL)
    for k in wanted:
        print(f"fig{k:02d}")
        t0 = time.time()
        FIGURES_ALL[k]()
        print(f"  {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv)
