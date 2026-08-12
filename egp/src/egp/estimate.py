"""Entropy rate estimation driver: SMB applied to particle-filter likelihoods."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .approx import entropy_rate_approx
from .pf import FilterResult, log_cell_prob, particle_filter_loglik
from .process import generate_quantized
from .spec import ProcessSpec

_LOG2 = np.log(2.0)


def marginal_entropy_bits(sigma: float, delta: float, max_symbols: int = 4_000_000) -> float:
    """H(Y_1) in bits for a single Gaussian sample rounded to a multiple of delta.

    Because conditioning cannot increase entropy this is a rigorous *upper*
    bound on the entropy rate of any quantized stationary process with this
    marginal.
    """
    sigma = float(sigma)
    delta = float(delta)
    reach = int(np.ceil(10.0 * sigma / delta)) + 1
    if 2 * reach + 1 > max_symbols:
        # Too fine to enumerate; the fine-quantization asymptote is exact here.
        return differential_entropy_bits(sigma) - np.log2(delta)
    y = np.arange(-reach, reach + 1, dtype=float)
    lo = (y - 0.5) * delta / sigma
    log_p = log_cell_prob(lo, lo + delta / sigma)
    p = np.exp(log_p)
    keep = p > 0
    return float(-np.sum(p[keep] * log_p[keep]) / _LOG2)


def differential_entropy_bits(scale: float) -> float:
    """Differential entropy of N(0, scale^2) in bits."""
    return float(0.5 * np.log2(2.0 * np.pi * np.e * float(scale) ** 2))


def fine_quantization_bits(h0: float, delta: float) -> float:
    """Fine-quantization value hbar - log2(delta), a rigorous *lower* bound.

    Since Y_1^n determines X_1^n only up to a box of side delta,
    h(X_1^n) = H(Y_1^n) + h(X_1^n | Y_1^n) <= H(Y_1^n) + n log delta, so the
    entropy rate is at least the Gaussian differential entropy rate less
    log2(delta).  It is also the leading asymptotics as delta / h0 -> 0.
    """
    return differential_entropy_bits(h0) - float(np.log2(delta))


def _stderr(values: np.ndarray) -> float:
    """Standard error of the mean; nan until there are two replicates."""
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(values.size))


def empirical_symbol_entropy_bits(y) -> float:
    """Order-0 (memoryless) plug-in entropy of the realized symbol stream."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


@dataclass
class RepeatUpdate:
    """One completed replicate, handed to ``on_repeat`` as the run proceeds."""

    index: int
    n_repeats: int
    entropy_rate_bits: float
    running_mean_bits: float  # over the usable replicates only; nan if none yet
    running_stderr_bits: float  # nan until the second usable replicate
    mean_ess: float
    lost_lock_fraction: float
    seconds: float
    #: False when this replicate lost the state, in which case its
    #: ``entropy_rate_bits`` is meaningless and excluded from the running mean.
    usable: bool = True


@dataclass
class EntropyRateEstimate:
    """Result of :func:`estimate_entropy_rate`, in bits per sample."""

    entropy_rate_bits: float
    stderr_bits: float
    per_repeat_bits: list[float]
    #: Per replicate: False where the filter lost the state.  Unusable
    #: replicates are excluded from ``entropy_rate_bits`` and ``stderr_bits``
    #: (their values are arbitrary, not noisy), but kept in
    #: ``per_repeat_bits`` so a run can be inspected.
    per_repeat_usable: list[bool]

    label: str
    delta: float
    sigma: float
    h0: float
    n_taps: int
    n: int
    n_particles: int
    n_repeats: int

    burn_in: int

    lower_bound_bits: float
    upper_bound_bits: float
    approximation_bits: float
    empirical_symbol_entropy_bits: float
    n_symbols_observed: int

    mean_ess: float
    min_ess: float
    lost_lock_fraction: float
    autocov_error: float
    floored_fraction: float
    elapsed_sec: float
    params: dict = field(default_factory=dict)

    @property
    def delta_over_sigma(self) -> float:
        return self.delta / self.sigma

    @property
    def delta_over_h0(self) -> float:
        return self.delta / self.h0

    @property
    def n_usable_repeats(self) -> int:
        return int(sum(self.per_repeat_usable))

    @property
    def n_failed_repeats(self) -> int:
        return self.n_repeats - self.n_usable_repeats

    @property
    def within_bounds(self) -> bool:
        return self.lower_bound_bits - 1e-9 <= self.entropy_rate_bits <= self.upper_bound_bits + 1e-9

    @property
    def collapsed(self) -> bool:
        """True when the run leaves no trustworthy estimate at all.

        Either every replicate lost the state, or what survived violates a
        bound that holds for the exact entropy rate.  Both mean the same
        remedy: more particles.  A run where only *some* replicates failed
        still reports a number — the mean over the survivors — but
        :attr:`n_failed_repeats` is then non-zero and the result should be
        treated as provisional, since the same particle count produced the
        failures.
        """
        return self.n_usable_repeats == 0 or not self.within_bounds

    def to_dict(self) -> dict:
        out = asdict(self)
        out["delta_over_sigma"] = self.delta_over_sigma
        out["delta_over_h0"] = self.delta_over_h0
        out["n_usable_repeats"] = self.n_usable_repeats
        out["n_failed_repeats"] = self.n_failed_repeats
        out["within_bounds"] = self.within_bounds
        out["collapsed"] = self.collapsed
        return out


def estimate_entropy_rate(
    spec: ProcessSpec,
    delta: float,
    *,
    n: int = 50_000,
    n_particles: int = 1_000,
    n_repeats: int = 3,
    burn_in: int | None = None,
    seed: int | None = 0,
    progress=None,
    on_repeat=None,
    engine: str = "cpu",
) -> EntropyRateEstimate:
    """Estimate the entropy rate of the quantized process, in bits per sample.

    Each repeat draws an independent sequence of length ``n`` and runs an
    independent particle filter over it; the reported value is the mean over
    the repeats that kept lock and the error bar is the standard error of that
    mean.  A repeat whose filter lost the state is discarded rather than
    averaged in — its log-likelihood is an arbitrary number, not a noisy one,
    and a single such repeat would otherwise swamp the mean by many orders of
    magnitude.  ``n_failed_repeats`` counts the discards and should be zero in
    a run worth quoting.

    ``engine="gpu"`` runs the filters on a WebGPU device via :mod:`.pf_gpu`
    (``pip install egp[gpu]``): the repeats of a batch run concurrently, one
    workgroup each, on identical sequences to the CPU engine's — only the
    filter's internal randomness differs, so the engines agree statistically.
    Repeats then complete in batches rather than one at a time; ``on_repeat``
    still fires once per repeat, in order.  The white process (L = 1) is
    exact and instant on the CPU regardless of ``engine``.

    The estimate is biased *upward* on two counts (Jensen's inequality applied
    to the unbiased particle likelihood, and the finite-n gap between
    H(Y_1^n)/n and its limit), so it should be treated as an upper estimate
    that tightens as ``n`` and ``n_particles`` grow.

    ``burn_in`` leading steps are filtered but excluded from the rate, which
    drops the transient in which H(Y_t | Y_1^{t-1}) is still falling toward its
    limit.  The default is 0: that transient is only O(L / n) of the average,
    and measurably smaller than the opposing drift from particle-cloud
    degradation, so burning in does not reliably tighten anything.  It is
    exposed for studying the transient itself.

    Always check ``result.collapsed`` before using the number, and
    ``result.n_failed_repeats`` after it: when too few particles are used the
    filter loses the state and the estimate is meaningless rather than merely
    noisy.
    """
    delta = float(delta)
    if burn_in is None:
        burn_in = 0
    if engine not in ("cpu", "gpu"):
        raise ValueError(f"engine must be 'cpu' or 'gpu', not {engine!r}")
    use_gpu = engine == "gpu" and spec.n_taps > 1
    seeds = np.random.SeedSequence(seed).spawn(n_repeats)
    started = time.time()

    per_repeat: list[float] = []
    usable_flags: list[bool] = []
    ess_means: list[float] = []
    ess_mins: list[float] = []
    lost: list[float] = []
    symbol_entropies: list[float] = []
    alphabet: set[int] = set()

    def usable_values() -> np.ndarray:
        return np.array(
            [v for v, ok in zip(per_repeat, usable_flags) if ok], dtype=float
        )

    def record(rep: int, result: FilterResult, seconds: float) -> None:
        per_repeat.append(result.entropy_rate_bits)
        usable_flags.append(not result.lost_lock)
        ess_means.append(result.mean_ess)
        ess_mins.append(result.min_ess)
        lost.append(result.lost_lock_fraction)
        if on_repeat is not None:
            # A lost-lock replicate contributes an arbitrary number, not a
            # noisy one, so the running mean is over the survivors only.
            so_far = usable_values()
            on_repeat(RepeatUpdate(
                index=rep,
                n_repeats=n_repeats,
                entropy_rate_bits=result.entropy_rate_bits,
                running_mean_bits=float(so_far.mean()) if so_far.size else float("nan"),
                running_stderr_bits=_stderr(so_far),
                mean_ess=result.mean_ess,
                lost_lock_fraction=result.lost_lock_fraction,
                seconds=seconds,
                usable=not result.lost_lock,
            ))

    def draw(rng: np.random.Generator) -> np.ndarray:
        _, y = generate_quantized(spec.taps, n, delta, rng)
        alphabet.update(np.unique(y).tolist())
        symbol_entropies.append(empirical_symbol_entropy_bits(y))
        return y

    if use_gpu:
        from .pf_gpu import max_batch, particle_filter_loglik_gpu

        batch_size = min(n_repeats, max_batch(n_particles, spec.n_taps, n))
        for first in range(0, n_repeats, batch_size):
            children = seeds[first : first + batch_size]
            batch_started = time.time()
            # Same generator, same seed as the CPU engine: identical sequences.
            ys = np.stack([draw(np.random.default_rng(child)) for child in children])
            gpu_seeds = np.array(
                [child.generate_state(1, dtype=np.uint32)[0] for child in children],
                dtype=np.uint32,
            )

            def report(done, total, _first=first):
                if progress is not None:
                    progress(_first, n_repeats, done, total)

            results = particle_filter_loglik_gpu(
                ys,
                spec.taps,
                delta,
                n_particles,
                gpu_seeds,
                burn_in=burn_in,
                progress=report if progress else None,
            )
            per_rep_sec = (time.time() - batch_started) / len(results)
            for offset, result in enumerate(results):
                record(first + offset, result, per_rep_sec)
    else:
        for rep, child in enumerate(seeds):
            rep_started = time.time()
            rng = np.random.default_rng(child)
            y = draw(rng)

            def report(done, total, _rep=rep):
                if progress is not None:
                    progress(_rep, n_repeats, done, total)

            result: FilterResult = particle_filter_loglik(
                y,
                spec.taps,
                delta,
                n_particles,
                rng,
                burn_in=burn_in,
                progress=report if progress else None,
            )
            record(rep, result, time.time() - rep_started)

    values = np.asarray(per_repeat, dtype=float)
    usable = usable_values()
    if usable.size:
        # Only the replicates that kept lock carry information; a lost-lock
        # run's likelihood is an arbitrary number that would swamp any mean.
        reported = float(usable.mean())
        stderr = _stderr(usable) if usable.size > 1 else 0.0
        ess_used = [e for e, ok in zip(ess_means, usable_flags) if ok]
        ess_min_used = [e for e, ok in zip(ess_mins, usable_flags) if ok]
    else:
        # Nothing usable: report the meaningless value anyway so the caller
        # can show what went wrong, with `collapsed` set.
        reported = float(values.mean())
        stderr = float("nan")
        ess_used, ess_min_used = ess_means, ess_mins

    return EntropyRateEstimate(
        entropy_rate_bits=reported,
        stderr_bits=stderr,
        per_repeat_bits=[float(v) for v in values],
        per_repeat_usable=list(usable_flags),
        label=spec.label,
        delta=delta,
        sigma=spec.sigma,
        h0=spec.h0,
        n_taps=spec.n_taps,
        n=n,
        n_particles=n_particles,
        n_repeats=n_repeats,
        burn_in=int(burn_in),
        lower_bound_bits=max(0.0, fine_quantization_bits(spec.h0, delta)),
        upper_bound_bits=marginal_entropy_bits(spec.sigma, delta),
        approximation_bits=entropy_rate_approx(spec.taps, delta),
        empirical_symbol_entropy_bits=float(np.mean(symbol_entropies)),
        n_symbols_observed=len(alphabet),
        mean_ess=float(np.mean(ess_used)),
        min_ess=float(np.min(ess_min_used)),
        # Averaged over *all* replicates: the honest global picture of how
        # often the filter lost the state, including the discarded runs.
        lost_lock_fraction=float(np.mean(lost)),
        autocov_error=spec.autocov_error,
        floored_fraction=spec.floored_fraction,
        elapsed_sec=time.time() - started,
        params={**spec.params, "engine": engine},
    )
