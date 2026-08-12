"""Correctness tests, including an independent brute-force likelihood check."""

from __future__ import annotations

import numpy as np
import pytest

import egp
from egp.pf import log_cell_prob, particle_filter_loglik, sample_truncnorm

LOG2 = np.log(2.0)


# --------------------------------------------------------------------------
# spectral factorization
# --------------------------------------------------------------------------


def _fake_result(bits: float, *, n_lost: int, n_steps: int = 1_000):
    """A FilterResult with a chosen entropy rate and lost-lock count."""
    from egp.pf import FilterResult

    loglik = -bits * n_steps * np.log(2.0)
    return FilterResult(
        loglik_nats=loglik,
        loglik_counted=loglik,
        n_steps=n_steps,
        burn_in=0,
        mean_ess=0.5,
        min_ess=0.1,
        n_lost_lock=n_lost,
        first_lost_step=0 if n_lost else -1,
        collapse_threshold=-25.0,
    )


def test_factorization_matches_target_autocovariance():
    rng = np.random.default_rng(1)
    kernel = rng.standard_normal(24)
    target = egp.autocovariance_from_taps(kernel)
    fac = egp.minimum_phase_from_autocov(target, n_taps=24)
    assert fac.autocov_error < 1e-9
    assert fac.h0 > 0


def test_boxcar_is_already_minimum_phase():
    # 1 + z^-1 + ... has all its roots on the unit circle, so factorization
    # should return the same filter (up to the tiny effect of the floor).
    kernel = np.ones(8)
    fac = egp.minimum_phase_from_taps(kernel)
    assert np.allclose(fac.taps, kernel, atol=1e-3)
    assert fac.autocov_error < 1e-6


def test_minimum_phase_beats_linear_phase_leading_tap():
    from scipy.signal import firwin

    kernel = firwin(65, 6000.0, fs=30000.0)
    fac = egp.minimum_phase_from_taps(kernel)
    # The whole point of factorizing: a symmetric design has a negligible
    # leading tap, which would be a degenerate prediction error variance.
    assert fac.taps[0] > 10 * abs(kernel[0])
    assert abs(fac.taps).argmax() < abs(kernel).argmax()  # energy moved to the front
    assert fac.autocov_error < 1e-8


def test_szego_matches_leading_tap():
    spec = egp.from_preset("ar1", rho=0.7, taps=200)
    gamma = spec.autocovariance()
    assert egp.szego_h0(gamma) == pytest.approx(spec.h0, rel=1e-4)


def test_ar1_leading_tap_is_prediction_error():
    rho = 0.8
    spec = egp.from_preset("ar1", rho=rho, taps=400, sigma=1.0)
    assert spec.sigma == pytest.approx(1.0)
    assert spec.h0 == pytest.approx(np.sqrt(1 - rho**2), rel=1e-3)


# --------------------------------------------------------------------------
# numerics of the cell probability / truncated sampler
# --------------------------------------------------------------------------


def test_log_cell_prob_deep_tails():
    from scipy.stats import norm

    lo = np.array([-1.0, 0.0, 3.0, -40.0, 38.0, -0.001])
    hi = lo + np.array([1.0, 0.5, 0.2, 0.05, 0.1, 0.002])
    got = log_cell_prob(lo, hi)
    with np.errstate(divide="ignore"):  # norm.cdf underflows in the deep tails
        want = np.log(norm.cdf(hi) - norm.cdf(lo))
    assert np.allclose(got[:3], want[:3], atol=1e-12)
    assert np.all(np.isfinite(got))  # far tails must not underflow to -inf

    # Deep tails, where norm.cdf underflows to give -inf.  Reference via the
    # scaled complementary error function, an independent code path that keeps
    # full relative precision arbitrarily far out.
    from scipy.special import erfcx

    def reference_one_sided(p, q):
        """log(Phi(q) - Phi(p)) for an interval on one side of the origin."""
        if p > 0:
            p, q = -q, -p
        assert q <= 0
        a, b = -q, -p  # 0 <= a <= b
        za, zb = a / np.sqrt(2.0), b / np.sqrt(2.0)
        bracket = 0.5 * (erfcx(za) - erfcx(zb) * np.exp(za**2 - zb**2))
        return -(za**2) + np.log(bracket)

    for k, (a, b) in enumerate(zip(lo, hi)):
        if a * b > 0 and min(abs(a), abs(b)) > 2:
            assert got[k] == pytest.approx(reference_one_sided(a, b), rel=1e-10), (k, a, b)

    # Symmetry: the standard normal cell probability is even.
    assert np.allclose(log_cell_prob(lo, hi), log_cell_prob(-hi, -lo))


def test_truncated_sampler_stays_in_cell_and_matches_moments():
    from scipy.stats import truncnorm

    rng = np.random.default_rng(3)
    for a, b in [(-0.5, 0.5), (1.0, 1.4), (-6.0, -5.9), (30.0, 30.2)]:
        lo = np.full(200_000, a)
        hi = np.full(200_000, b)
        x = sample_truncnorm(lo, hi, rng)
        assert np.all(x >= a) and np.all(x <= b)
        want = truncnorm.mean(a, b)
        assert x.mean() == pytest.approx(want, abs=5e-3 * max(1.0, abs(want)))


# --------------------------------------------------------------------------
# the particle filter against an independent brute-force likelihood
# --------------------------------------------------------------------------


def exact_loglik_L2(y, h, delta, n_grid=6001, span=10.0):
    """log P(y_1..y_n) for a 2-tap process by deterministic quadrature.

    The hidden state is the single innovation W_{t-1}, so the filtering
    recursion is a one-dimensional integral that can be done on a grid:

        alpha_t(w_t) = phi(w_t) * Int alpha_{t-1}(v) 1{h0 w_t + h1 v in C_yt} dv

    The inner integral is evaluated from the cumulative trapezoid of
    alpha_{t-1}, which keeps the error second order in the grid spacing rather
    than first order as a hard indicator mask would.
    """
    h0, h1 = float(h[0]), float(h[1])
    assert h1 != 0.0
    w = np.linspace(-span, span, n_grid)
    alpha = np.exp(-0.5 * w**2) / np.sqrt(2 * np.pi)
    phi = alpha.copy()
    log_scale = 0.0

    for yt in np.asarray(y):
        lo = (yt - 0.5) * delta
        hi = lo + delta
        # v must satisfy lo <= h0*w + h1*v < hi
        edge_a = (lo - h0 * w) / h1
        edge_b = (hi - h0 * w) / h1
        left = np.minimum(edge_a, edge_b)
        right = np.maximum(edge_a, edge_b)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (alpha[1:] + alpha[:-1]) * np.diff(w))])
        mass = np.interp(right, w, cum, left=0.0, right=cum[-1]) - np.interp(
            left, w, cum, left=0.0, right=cum[-1]
        )
        alpha = phi * mass
        peak = alpha.max()
        if peak <= 0:
            return -np.inf
        alpha /= peak
        log_scale += np.log(peak)

    total = np.trapezoid(alpha, w) if hasattr(np, "trapezoid") else np.trapz(alpha, w)
    return float(np.log(total) + log_scale)


def test_particle_filter_matches_exact_likelihood_two_taps():
    h = np.array([0.8, 0.6])
    delta = 1.0
    rng = np.random.default_rng(11)
    _, y = egp.generate_quantized(h, 8, delta, rng)

    reference = exact_loglik_L2(y, h, delta)

    values = []
    for seed in range(6):
        res = particle_filter_loglik(y, h, delta, 40_000, np.random.default_rng(100 + seed))
        values.append(res.loglik_nats)
    got = float(np.mean(values))
    spread = float(np.std(values, ddof=1) / np.sqrt(len(values)))

    assert abs(got - reference) < max(5 * spread, 0.02), (got, reference, spread)


def test_particle_filter_matches_exact_likelihood_fine_quantization():
    h = np.array([0.6, 0.8])  # deliberately not minimum phase in magnitude order
    h = egp.minimum_phase_from_taps(h).taps
    delta = 0.2
    rng = np.random.default_rng(12)
    _, y = egp.generate_quantized(h, 6, delta, rng)

    reference = exact_loglik_L2(y, h, delta)
    values = [
        particle_filter_loglik(y, h, delta, 40_000, np.random.default_rng(200 + s)).loglik_nats
        for s in range(6)
    ]
    got = float(np.mean(values))
    spread = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    assert abs(got - reference) < max(5 * spread, 0.02), (got, reference, spread)


# --------------------------------------------------------------------------
# entropy rate
# --------------------------------------------------------------------------


def test_white_noise_entropy_rate_is_exact():
    # L = 1: the symbols are i.i.d., the filter is exact, and the answer is the
    # closed-form marginal entropy.
    spec = egp.from_preset("white")
    delta = 0.7
    want = egp.marginal_entropy_bits(spec.sigma, delta)
    got = egp.estimate_entropy_rate(
        spec, delta, n=400_000, n_particles=2, n_repeats=2, seed=5
    )
    assert got.entropy_rate_bits == pytest.approx(want, abs=0.01)
    assert got.upper_bound_bits == pytest.approx(want, abs=1e-9)


def test_fine_quantization_approaches_gaussian_asymptote():
    spec = egp.from_preset("ar1", rho=0.8, taps=64)
    delta = 0.02  # delta / h0 = 0.033
    res = egp.estimate_entropy_rate(spec, delta, n=20_000, n_particles=400, n_repeats=2, seed=7)
    want = egp.fine_quantization_bits(spec.h0, delta)
    assert res.entropy_rate_bits == pytest.approx(want, abs=0.03)


def test_estimate_respects_rigorous_bounds():
    for preset, kwargs, delta in [
        ("ma", dict(taps=8), 0.5),
        ("ar1", dict(rho=0.9, taps=32), 1.0),
        ("gaussian", dict(tau=1.5, taps=9), 0.5),
    ]:
        spec = egp.from_preset(preset, **kwargs)
        res = egp.estimate_entropy_rate(
            spec, delta, n=5_000, n_particles=1_000, n_repeats=1, seed=13
        )
        assert not res.collapsed, (preset, res.entropy_rate_bits, res.lost_lock_fraction)
        assert res.within_bounds, (preset, res.entropy_rate_bits)


def test_collapse_is_detected_not_silently_reported():
    # A sharp lowpass has h0 far below sigma, so a small cloud loses the state
    # in a 32-dimensional space.  The failure must be flagged, never returned
    # as a plausible-looking number.
    spec = egp.from_preset("lowpass", cutoff=6000.0, fs=30000.0, taps=33)
    bad = egp.estimate_entropy_rate(spec, 0.25, n=3_000, n_particles=300, n_repeats=1, seed=13)
    assert bad.collapsed
    assert bad.lost_lock_fraction > 0.5


def test_lost_lock_replicates_are_excluded_from_the_average(monkeypatch):
    # A replicate that loses the state returns an arbitrary number, not a noisy
    # one, so averaging it in would move the answer by orders of magnitude.
    # Substitute the filter to produce a known good/bad/good sequence.
    import egp.estimate as estimate_mod

    canned = [
        _fake_result(2.0, n_lost=0),
        _fake_result(1.7e7, n_lost=900),  # lost lock: garbage value
        _fake_result(2.2, n_lost=0),
    ]
    calls = iter(canned)
    monkeypatch.setattr(estimate_mod, "particle_filter_loglik", lambda *a, **k: next(calls))

    seen = []
    res = egp.estimate_entropy_rate(
        egp.from_preset("ma", taps=4), 0.5, n=1_000, n_particles=100,
        n_repeats=3, seed=0, on_repeat=seen.append,
    )

    assert res.per_repeat_bits == [2.0, 1.7e7, 2.2]
    assert res.per_repeat_usable == [True, False, True]
    assert res.n_failed_repeats == 1 and res.n_usable_repeats == 2
    # The mean is over the survivors only, so the garbage value is nowhere in it.
    assert res.entropy_rate_bits == pytest.approx(2.1)
    assert res.stderr_bits == pytest.approx(0.1)
    # Something failed, but a usable estimate survived: not a total collapse.
    assert not res.collapsed
    # The streamed running mean excludes it too, and never reports the garbage.
    assert [u.usable for u in seen] == [True, False, True]
    assert seen[0].running_mean_bits == pytest.approx(2.0)
    assert seen[1].running_mean_bits == pytest.approx(2.0)  # unchanged by the failure
    assert seen[2].running_mean_bits == pytest.approx(2.1)


def test_every_replicate_failing_is_a_collapse(monkeypatch):
    import egp.estimate as estimate_mod

    monkeypatch.setattr(
        estimate_mod, "particle_filter_loglik", lambda *a, **k: _fake_result(5e6, n_lost=999)
    )
    res = egp.estimate_entropy_rate(
        egp.from_preset("ma", taps=4), 0.5, n=1_000, n_particles=100, n_repeats=2, seed=0
    )
    assert res.collapsed
    assert res.n_usable_repeats == 0 and res.n_failed_repeats == 2
    assert np.isnan(res.stderr_bits)  # no error bar without a usable replicate


def test_collapse_threshold_scales_with_cell_width():
    from egp.pf import collapse_threshold

    # Under fine quantization every step is legitimately worth log(delta/h0)
    # nats, so the threshold has to move down with it.
    assert collapse_threshold(1.0, 0.1) == pytest.approx(-25.0)  # delta/h0 = 10 -> capped at 1
    assert collapse_threshold(0.001, 1.0) == pytest.approx(np.log(0.001) - 25.0)


def test_burn_in_excludes_exactly_the_leading_steps():
    # Same seed means identical particle trajectories, so burn-in may only
    # change which steps are summed -- never the filtering itself.
    h = egp.from_preset("ar1", rho=0.9, taps=16).taps
    rng = np.random.default_rng(21)
    _, y = egp.generate_quantized(h, 500, 0.5, rng)

    full = particle_filter_loglik(y, h, 0.5, 500, np.random.default_rng(1), burn_in=0)
    part = particle_filter_loglik(y, h, 0.5, 500, np.random.default_rng(1), burn_in=100)

    assert full.loglik_nats == pytest.approx(part.loglik_nats, rel=1e-12)
    assert full.loglik_counted == pytest.approx(full.loglik_nats, rel=1e-12)
    assert part.n_counted == 400
    head = full.loglik_nats - part.loglik_counted  # the excluded first 100 steps
    assert head < 0  # log-probabilities

    # Default is no burn-in: the transient it removes is O(L / n).
    assert egp.estimate_entropy_rate(
        egp.from_preset("ma", taps=4), 0.5, n=2000, n_particles=200, n_repeats=1
    ).burn_in == 0


def test_correlation_reduces_entropy_rate_below_memoryless():
    spec = egp.from_preset("ma", taps=8)
    res = egp.estimate_entropy_rate(spec, 0.5, n=20_000, n_particles=500, n_repeats=1, seed=2)
    assert res.entropy_rate_bits < res.empirical_symbol_entropy_bits - 0.1


def test_on_repeat_fires_once_per_replicate_with_running_statistics():
    seen = []
    res = egp.estimate_entropy_rate(
        egp.from_preset("ma", taps=4), 0.5, n=2_000, n_particles=200,
        n_repeats=3, seed=1, on_repeat=seen.append,
    )
    assert [u.index for u in seen] == [0, 1, 2]
    assert all(u.n_repeats == 3 for u in seen)
    assert [u.entropy_rate_bits for u in seen] == res.per_repeat_bits

    # The running mean is the mean of the replicates reported so far, and the
    # last update agrees with the final result.
    for k, u in enumerate(seen):
        assert u.running_mean_bits == pytest.approx(np.mean(res.per_repeat_bits[: k + 1]))
    assert np.isnan(seen[0].running_stderr_bits)  # undefined from one replicate
    assert seen[-1].running_mean_bits == pytest.approx(res.entropy_rate_bits)
    assert seen[-1].running_stderr_bits == pytest.approx(res.stderr_bits)
    assert all(u.seconds >= 0 for u in seen)


def test_cli_estimate_streams_each_repeat(capsys):
    from egp.cli import main

    assert main(["estimate", "--preset", "ma", "--taps", "4", "--delta", "0.5",
                 "-n", "2000", "-N", "200", "-r", "3"]) == 0
    cap = capsys.readouterr()
    lines = [ln for ln in cap.err.splitlines() if "bits/sample" in ln]
    assert len(lines) == 3
    assert lines[0].startswith("  [1/3]") and lines[2].startswith("  [3/3]")
    assert "+/-      --" in lines[0]  # no standard error from one replicate yet
    # streamed to stderr, leaving stdout for the result
    assert "bits/sample" in cap.out and "[1/3]" not in cap.out


def test_more_particles_does_not_increase_the_estimate():
    spec = egp.from_preset("ma", taps=6)
    small = egp.estimate_entropy_rate(spec, 0.4, n=20_000, n_particles=20, n_repeats=3, seed=4)
    large = egp.estimate_entropy_rate(spec, 0.4, n=20_000, n_particles=2000, n_repeats=3, seed=4)
    assert large.entropy_rate_bits <= small.entropy_rate_bits + 1e-3


# --------------------------------------------------------------------------
# process / spec plumbing
# --------------------------------------------------------------------------


def test_generated_process_has_the_requested_autocovariance():
    spec = egp.from_preset("lowpass", cutoff=0.1, fs=1.0, taps=33)
    rng = np.random.default_rng(0)
    x = egp.generate(spec.taps, 400_000, rng)
    want = spec.autocovariance(max_lag=5)
    got = np.array([np.mean(x[: x.size - k] * x[k:]) for k in range(6)])
    assert np.allclose(got, want, atol=0.02)


def test_quantize_rounds_to_nearest_multiple():
    x = np.array([-0.6, -0.4, 0.0, 0.4, 0.6, 1.49, 1.51])
    assert np.array_equal(egp.quantize(x, 1.0), [-1, 0, 0, 0, 1, 1, 2])


def test_all_presets_build():
    for name in egp.PRESETS:
        spec = egp.from_preset(name, cutoff=0.2, band=(0.1, 0.3))
        assert spec.h0 > 0
        assert spec.sigma == pytest.approx(1.0)


def test_custom_autocovariance_roundtrip():
    lags = np.arange(40)
    gamma = 0.85**lags
    spec = egp.from_autocov(gamma, n_taps=120)
    assert spec.autocov_error < 1e-6
    assert spec.h0 == pytest.approx(np.sqrt(1 - 0.85**2), rel=0.02)


def test_cli_estimate_and_plot(tmp_path, capsys):
    from egp.cli import main

    rc = main(
        [
            "estimate", "--preset", "ma", "--taps", "4", "--delta", "0.5",
            "-n", "2000", "-N", "200", "-r", "1", "--quiet",
            "--json", str(tmp_path / "out.json"),
        ]
    )
    assert rc == 0
    assert "entropy rate" in capsys.readouterr().out
    assert (tmp_path / "out.json").exists()

    rc = main(
        [
            "plot", "--preset", "lowpass", "--cutoff", "6000", "--fs", "30000",
            "--taps", "33", "--delta", "0.5", "--n", "20000",
            "-o", str(tmp_path / "fig.png"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "fig.png").stat().st_size > 10_000
