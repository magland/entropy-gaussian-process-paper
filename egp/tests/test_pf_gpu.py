"""GPU engine tests: skipped wholesale when wgpu or an adapter is missing.

The GPU filter shares no RNG with the CPU filter, so the checks are the
statistical ones that pin any correct implementation: agreement with the CPU
engine on identical sequences to within Monte-Carlo error, near-identical ESS
(a property of the model, not the randomness), and consistency through
``estimate_entropy_rate(engine="gpu")``.
"""

import numpy as np
import pytest

import egp

pytestmark = pytest.mark.skipif(not egp.gpu_available(), reason="no wgpu / WebGPU adapter")

N_PARTICLES = 2_000
N = 4_000
R = 4


@pytest.fixture(scope="module")
def ma8():
    return egp.from_preset("ma", taps=8, sigma=5.0)


def _cpu_reference(spec, ys, delta):
    results = [
        egp.particle_filter_loglik(y, spec.taps, delta, N_PARTICLES, np.random.default_rng(50 + r))
        for r, y in enumerate(ys)
    ]
    bits = np.array([r.entropy_rate_bits for r in results])
    return bits, np.mean([r.mean_ess for r in results])


def test_gpu_matches_cpu_on_identical_sequences(ma8):
    delta = 1.0
    rng = np.random.default_rng(3)
    ys = [egp.generate_quantized(ma8.taps, N, delta, rng)[1] for _ in range(R)]

    gpu = egp.particle_filter_loglik_gpu(
        np.stack(ys), ma8.taps, delta, N_PARTICLES, np.arange(1, R + 1, dtype=np.uint32)
    )
    gpu_bits = np.array([r.entropy_rate_bits for r in gpu])
    cpu_bits, cpu_ess = _cpu_reference(ma8, ys, delta)

    # Identical data, independent filter noise: the difference is pure filter
    # Monte-Carlo error, ~se/sqrt(R) per mean with rep sd ~0.01 bits here.
    assert abs(gpu_bits.mean() - cpu_bits.mean()) < 0.05
    # ESS is a property of the model; it reproduces closely across engines.
    assert abs(np.mean([r.mean_ess for r in gpu]) - cpu_ess) < 0.03
    for r in gpu:
        assert r.lost_lock_fraction < 1e-3
        assert r.n_steps == N


def test_gpu_burn_in_counted_consistently(ma8):
    delta = 1.0
    rng = np.random.default_rng(9)
    y = egp.generate_quantized(ma8.taps, N, delta, rng)[1]
    burn = 500
    (res,) = egp.particle_filter_loglik_gpu(
        y[None, :], ma8.taps, delta, N_PARTICLES, np.array([7], dtype=np.uint32), burn_in=burn
    )
    assert res.burn_in == burn
    assert res.n_counted == N - burn
    # The counted loglik is the total minus the burn-in prefix, by construction.
    assert res.loglik_counted < 0
    assert res.loglik_counted > res.loglik_nats


def test_estimate_entropy_rate_gpu_engine(ma8):
    updates = []
    result = egp.estimate_entropy_rate(
        ma8,
        1.0,
        n=N,
        n_particles=N_PARTICLES,
        n_repeats=3,
        seed=1,
        engine="gpu",
        on_repeat=updates.append,
    )
    assert not result.collapsed
    assert len(updates) == 3
    assert [u.index for u in updates] == [0, 1, 2]
    assert result.params["engine"] == "gpu"
    # Within a few Monte-Carlo standard errors of the analytic approximation.
    assert abs(result.entropy_rate_bits - result.approximation_bits) < 0.1


def test_white_process_falls_back_to_exact_cpu():
    spec = egp.from_preset("white", sigma=5.0)
    result = egp.estimate_entropy_rate(
        spec, 1.0, n=20_000, n_particles=10, n_repeats=2, seed=0, engine="gpu"
    )
    # L = 1 is exact regardless of engine; it must land on H(Y_1).
    assert abs(result.entropy_rate_bits - result.upper_bound_bits) < 0.05
