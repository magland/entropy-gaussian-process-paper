"""Fully adapted particle filter for the log-likelihood of a quantized sample.

Implements Section 4.3 of the paper.  The hidden state is the
window of the last L-1 innovations; given that window, X_t is N(mu, h0^2), so
the observation weight is the Gaussian cell probability Phi(r) - Phi(l) and the
optimal proposal for the new innovation is N(0, 1) truncated to [l, r).  Both
are available in closed form, which makes the filter fully adapted: resampling
uses the exact predictive weights and no importance weights survive the step.

All internal quantities are in nats; callers convert to bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr

try:  # SciPy >= 1.8
    from scipy.special import ndtri_exp
except ImportError:  # pragma: no cover - fallback for very old SciPy
    from scipy.special import ndtri

    def ndtri_exp(logp):
        return ndtri(np.exp(logp))


_LOG2 = np.log(2.0)


def _log1mexp(x) -> np.ndarray:
    """log(1 - exp(x)) for x <= 0, accurate at both ends."""
    x = np.asarray(x, dtype=float)
    flat = np.atleast_1d(x)
    out = np.empty_like(flat)
    near = flat > -_LOG2
    with np.errstate(divide="ignore", invalid="ignore"):
        out[near] = np.log(-np.expm1(flat[near]))
        far = ~near
        out[far] = np.log1p(-np.exp(flat[far]))
    return out.reshape(x.shape)


def _cell_stats(lo: np.ndarray, hi: np.ndarray):
    """Shared work behind both the weight and the proposal.

    Returns ``(log_prob, log_lower_cdf, mirror)`` where ``log_prob`` is
    ``log(Phi(hi) - Phi(lo))`` and the other two are what the inverse-CDF draw
    needs.  Reflecting an interval whose midpoint is positive keeps the pair of
    endpoints on the side where ``log_ndtr`` retains relative precision; without
    it, two large positive endpoints both give ``log_ndtr ~ 0`` and their
    difference underflows.
    """
    mirror = (lo + hi) > 0
    a = np.where(mirror, -hi, lo)
    b = np.where(mirror, -lo, hi)
    la = log_ndtr(a)
    lb = log_ndtr(b)
    return lb + _log1mexp(la - lb), la, mirror


def log_cell_prob(lo, hi) -> np.ndarray:
    """log(Phi(hi) - Phi(lo)) for hi >= lo, stable deep into either tail."""
    return _cell_stats(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))[0]


def _draw(log_mass, la, mirror, lo, hi, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF draw from N(0, 1) on [lo, hi) given precomputed cell stats."""
    u = rng.random(la.shape)
    with np.errstate(divide="ignore"):
        log_p = np.logaddexp(la, np.log(u) + log_mass)
    x = ndtri_exp(log_p)
    x = np.where(mirror, -x, x)
    bad = ~np.isfinite(x)
    if bad.any():  # numerically empty cell: fall back to its midpoint
        x = np.where(bad, 0.5 * (lo + hi), x)
    return np.clip(x, lo, hi)


def sample_truncnorm(lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw N(0, 1) truncated to [lo, hi) by inverse CDF in log space."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    log_mass, la, mirror = _cell_stats(lo, hi)
    return _draw(log_mass, la, mirror, lo, hi, rng)


DEFAULT_COLLAPSE_MARGIN = 25.0

#: Fraction of lost-lock steps above which a replicate is unusable.  A filter
#: that is tracking has none at all, so this is a wide margin around zero
#: rather than a tuned cutoff.
LOST_LOCK_TOLERANCE = 1e-3


@dataclass
class FilterResult:
    loglik_nats: float  # over all n steps
    loglik_counted: float  # over the steps after burn-in
    n_steps: int
    burn_in: int
    mean_ess: float  # averaged effective sample size, as a fraction of N
    min_ess: float
    n_lost_lock: int
    first_lost_step: int  # -1 if the filter never lost lock
    collapse_threshold: float

    @property
    def n_counted(self) -> int:
        return self.n_steps - self.burn_in

    @property
    def entropy_rate_bits(self) -> float:
        if self.n_counted <= 0:
            return float("nan")
        return -self.loglik_counted / (self.n_counted * _LOG2)

    @property
    def lost_lock_fraction(self) -> float:
        return self.n_lost_lock / self.n_steps if self.n_steps else 0.0

    @property
    def lost_lock(self) -> bool:
        """True when the particle cloud lost the state on this run.

        The log-likelihood of such a run is not a noisy estimate of the true
        one but an arbitrary number, so callers must discard it rather than
        average it in.
        """
        return self.lost_lock_fraction > LOST_LOCK_TOLERANCE


def collapse_threshold(delta: float, h0: float, margin: float = DEFAULT_COLLAPSE_MARGIN) -> float:
    """Per-step log-likelihood below which the filter has certainly lost the state.

    A particle sitting ``d`` prediction standard deviations away from the
    observed cell contributes about ``log(delta / h0) - d^2 / 2``, so a filter
    that is tracking at all cannot fall far below ``log(min(1, delta / h0))``.
    Anything ``margin`` nats under that is a lost-lock step, not an unlucky
    symbol.  The scale term matters: under fine quantization *every* step is
    legitimately worth about ``log(delta / h0)`` nats.
    """
    return float(np.log(min(1.0, delta / h0)) - margin)


def particle_filter_loglik(
    y,
    taps,
    delta: float,
    n_particles: int,
    rng: np.random.Generator,
    burn_in: int = 0,
    collapse_margin: float = DEFAULT_COLLAPSE_MARGIN,
    progress=None,
) -> FilterResult:
    """Estimate log P(y_1..y_n) for the quantized FIR process.

    Parameters
    ----------
    y : integer array
        Observed symbols, ``y_t = round(x_t / delta)``.
    taps : array
        Minimum-phase FIR taps; ``taps[0]`` must be positive.
    delta : float
        Quantization step.
    n_particles : int
        Number of particles N.  Ignored when the process is white (L = 1), for
        which the filter is exact.
    burn_in : int
        Number of leading steps excluded from ``loglik_counted``.  The filter
        still runs them, so the excluded part is what lets the particle cloud
        lock onto the state; dropping it estimates
        ``-log P(y_{B+1..n} | y_1..y_B) / (n - B)``, which is still an upper
        bound on the entropy rate but a tighter one.
    collapse_margin : float
        Nats below :func:`collapse_threshold` at which a step counts as
        lost-lock.  See :func:`collapse_threshold`.
    progress : callable, optional
        Called as ``progress(t, n)`` every ~1% of the sweep.
    """
    y = np.asarray(y, dtype=np.int64).ravel()
    taps = np.asarray(taps, dtype=float).ravel()
    delta = float(delta)
    n = y.size
    n_taps = taps.size
    h0 = float(taps[0])
    if not h0 > 0:
        raise ValueError("taps[0] must be positive (use the minimum-phase factorization)")
    if delta <= 0:
        raise ValueError("delta must be positive")
    burn_in = int(np.clip(burn_in, 0, max(0, n - 1)))
    limit = collapse_threshold(delta, h0, collapse_margin)
    if n == 0:
        return FilterResult(0.0, 0.0, 0, 0, 1.0, 1.0, 0, -1, limit)

    lower = (y - 0.5) * delta
    width = delta / h0

    if n_taps == 1:  # white: symbols are i.i.d., the likelihood is exact
        lo = lower / h0
        steps = log_cell_prob(lo, lo + width)
        return FilterResult(
            loglik_nats=float(np.sum(steps)),
            loglik_counted=float(np.sum(steps[burn_in:])),
            n_steps=n,
            burn_in=burn_in,
            mean_ess=1.0,
            min_ess=1.0,
            n_lost_lock=int(np.sum(steps < limit)),
            first_lost_step=int(np.argmax(steps < limit)) if np.any(steps < limit) else -1,
            collapse_threshold=limit,
        )

    n_particles = int(n_particles)
    if n_particles < 2:
        raise ValueError("n_particles must be >= 2")

    ring_len = n_taps - 1
    ring = rng.standard_normal((n_particles, ring_len))
    scratch = np.empty_like(ring)
    tap_buf = np.empty(ring_len)
    offsets = np.arange(ring_len)
    particle_index = np.arange(n_particles)
    tail = taps[1:]
    pos = 0

    loglik = 0.0
    loglik_counted = 0.0
    ess_sum = 0.0
    ess_min = float(n_particles)
    n_lost = 0
    first_lost = -1
    log_n = np.log(n_particles)
    report_every = max(1, n // 100)

    for t in range(n):
        # Rolled tap vector so that ring[:, (pos - k) % ring_len] holds W_{t-1-k}.
        tap_buf[(pos - offsets) % ring_len] = tail
        mu = ring @ tap_buf

        lo = (lower[t] - mu) / h0
        hi = lo + width
        # The weight and the optimal proposal share all their transcendentals.
        log_alpha, log_lower, mirror = _cell_stats(lo, hi)

        peak = log_alpha.max()
        if not np.isfinite(peak):
            raise RuntimeError(
                f"particle filter collapsed at step {t}: every particle assigns "
                "zero probability to the observed symbol. Increase --particles."
            )
        weights = np.exp(log_alpha - peak)
        w_sum = float(weights.sum())
        step = peak + np.log(w_sum) - log_n
        loglik += step
        if t >= burn_in:
            loglik_counted += step
        if step < limit:
            n_lost += 1
            if first_lost < 0:
                first_lost = t

        ess = w_sum * w_sum / float(np.dot(weights, weights))
        ess_sum += ess
        ess_min = min(ess_min, ess)

        # Systematic resampling with the exact predictive weights.
        cumulative = np.cumsum(weights)
        cumulative /= cumulative[-1]
        u = (rng.random() + particle_index) / n_particles
        ancestors = np.searchsorted(cumulative, u)
        np.clip(ancestors, 0, n_particles - 1, out=ancestors)
        np.take(ring, ancestors, axis=0, out=scratch)
        ring, scratch = scratch, ring

        # Propagate: the optimal proposal is the truncated innovation.
        w_new = _draw(
            log_alpha[ancestors],
            log_lower[ancestors],
            mirror[ancestors],
            lo[ancestors],
            hi[ancestors],
            rng,
        )
        pos = (pos + 1) % ring_len
        ring[:, pos] = w_new

        if progress is not None and (t % report_every == 0 or t == n - 1):
            progress(t + 1, n)

    return FilterResult(
        loglik_nats=float(loglik),
        loglik_counted=float(loglik_counted),
        n_steps=n,
        burn_in=burn_in,
        mean_ess=ess_sum / (n * n_particles),
        min_ess=ess_min / n_particles,
        n_lost_lock=n_lost,
        first_lost_step=first_lost,
        collapse_threshold=limit,
    )
