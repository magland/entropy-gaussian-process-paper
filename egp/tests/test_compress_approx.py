import bz2
import importlib.util
import json
import lzma
import math
import zlib

import numpy as np
import pytest

import egp
from egp.cli import main


@pytest.fixture(scope="module")
def correlated():
    spec = egp.from_preset("ma", taps=8)
    return egp.quantized_sample(spec.taps, 0.5, 20_000, np.random.default_rng(0))


# --------------------------------------------------------------------------
# linear prediction
# --------------------------------------------------------------------------


def test_lpc_round_trip_is_lossless(correlated):
    y = correlated[:2000]
    for order in (1, 4, 16):
        residual, model = egp.lpc_transform(y, order)
        assert model.order == order
        assert np.array_equal(egp.lpc_inverse(residual, model), y)


def test_lpc_preserves_the_leading_samples(correlated):
    residual, _ = egp.lpc_transform(correlated[:500], 8)
    assert np.array_equal(residual[:8], correlated[:8])


def test_lpc_reduces_residual_entropy(correlated):
    residual, _ = egp.lpc_transform(correlated, 8)
    assert egp.empirical_entropy_bits(residual) < egp.empirical_entropy_bits(correlated) - 0.5
    assert residual.std() < correlated.std()


def test_lpc_order_zero_is_the_identity(correlated):
    residual, model = egp.lpc_transform(correlated, 0)
    assert model.order == 0 and model.header_bytes == 1
    assert np.array_equal(residual, correlated)


def test_lpc_on_white_noise_barely_helps():
    y = egp.quantized_sample(np.ones(1), 0.5, 20_000, np.random.default_rng(1))
    residual, _ = egp.lpc_transform(y, 8)
    assert egp.empirical_entropy_bits(residual) > egp.empirical_entropy_bits(y) - 0.1


def test_delta_round_trip_and_reduces_entropy(correlated):
    d = egp.delta_transform(correlated)
    assert np.array_equal(egp.delta_inverse(d), correlated)
    assert d[0] == correlated[0]
    assert egp.empirical_entropy_bits(d) < egp.empirical_entropy_bits(correlated)


# --------------------------------------------------------------------------
# sampling and codecs
# --------------------------------------------------------------------------


def test_quantized_sample_matches_the_process():
    spec = egp.from_preset("ma", taps=4)
    y = egp.quantized_sample(spec.taps, 0.5, 200_000, np.random.default_rng(2))
    assert abs(egp.empirical_entropy_bits(y) - egp.marginal_entropy_bits(1.0, 0.5)) < 0.02
    assert abs(np.std(y * 0.5) / spec.sigma - 1.0) < 0.05


def test_quantized_sample_rejects_values_outside_int16():
    with pytest.raises(ValueError, match="int16"):
        egp.quantized_sample(np.ones(1), 1e-5, 1000, np.random.default_rng(0))


@pytest.mark.parametrize(
    "compress,decompress",
    [
        (lambda b: zlib.compress(b, 9), zlib.decompress),
        (lambda b: bz2.compress(b, 9), bz2.decompress),
        (lambda b: lzma.compress(b, preset=9), lzma.decompress),
    ],
)
def test_byte_codecs_round_trip(correlated, compress, decompress):
    payload = np.asarray(correlated, dtype="<i2").tobytes()
    restored = np.frombuffer(decompress(compress(payload)), dtype="<i2")
    assert np.array_equal(restored, correlated)


def test_ans_round_trip_and_near_entropy():
    simple_ans = pytest.importorskip("simple_ans")
    y = egp.quantized_sample(np.ones(1), 0.5, 100_000, np.random.default_rng(3))
    encoded = simple_ans.ans_encode(np.asarray(y, dtype=np.int16))
    assert np.array_equal(simple_ans.ans_decode(encoded), y)
    assert 8.0 * encoded.size() / y.size - egp.empirical_entropy_bits(y) < 0.05


def test_registry_has_the_standard_library_codecs():
    assert {"zlib-9", "bz2-9", "lzma-9"} <= set(egp.codec_registry())


def test_flac_encodes_losslessly_and_beats_the_raw_stream(correlated):
    pytest.importorskip("soundfile")  # also the decoder for the round trip
    import io

    import soundfile as sf

    n_bytes = egp.flac_size(correlated)
    assert 0 < 8.0 * n_bytes / correlated.size < 16.0

    buf = io.BytesIO()
    with sf.SoundFile(buf, "w", samplerate=egp.compression.FLAC_RATE, channels=1,
                      format="FLAC", subtype="PCM_16") as f:
        value = sf._ffi.new("double*", 1.0)
        sf._snd.sf_command(f._file, 0x1301, value, sf._ffi.sizeof("double"))
        f.write(np.asarray(correlated, dtype=np.int16))
    assert len(buf.getvalue()) == n_bytes
    buf.seek(0)
    restored, _ = sf.read(buf, dtype="int16")
    assert np.array_equal(restored, np.asarray(correlated, dtype=np.int16))


def test_flac_is_in_the_registry_when_a_backend_is_installed():
    have_backend = True
    try:
        import pyflac  # noqa: F401
    except ImportError:
        have_backend = importlib.util.find_spec("soundfile") is not None
    assert ("flac-8" in egp.codec_registry()) == have_backend


def test_codecs_refuse_values_outside_int16():
    big = np.array([0, 40_000, -40_000], dtype=np.int64)
    for name, fn in egp.codec_registry().items():
        with pytest.raises(ValueError, match="int16"):
            fn(big)


def test_benchmark_covers_both_transforms_and_sorts(correlated):
    results = egp.compress_benchmark(correlated, lpc_order=8, methods=["zlib-9", "lzma-9"])
    assert {r.transform for r in results} == {"raw", "lpc(8)"}
    assert {r.method for r in results} == {"zlib-9", "lzma-9"}
    bits = [r.bits_per_sample for r in results]
    assert bits == sorted(bits)
    for r in results:
        assert r.n_bytes > 0 and r.bits_per_sample < 16.0
        assert abs(r.ratio - 16.0 / r.bits_per_sample) < 1e-9
        assert abs(r.bits_per_sample - 8.0 * r.n_bytes / correlated.size) < 1e-9


def test_benchmark_without_lpc_has_one_transform(correlated):
    assert [r.transform for r in egp.compress_benchmark(
        correlated, lpc_order=0, methods=["zlib-9"])] == ["raw"]


def test_benchmark_runs_every_requested_transform(correlated):
    results = egp.compress_benchmark(
        correlated, lpc_order=8, transforms=("raw", "delta", "lpc"), methods=["zlib-9"]
    )
    assert {r.transform for r in results} == {"raw", "delta", "lpc(8)"}
    bits = {r.transform: r.bits_per_sample for r in results}
    assert bits["lpc(8)"] < bits["delta"] < bits["raw"]


def test_benchmark_rejects_unknown_codecs(correlated):
    with pytest.raises(ValueError, match="unknown or unavailable"):
        egp.compress_benchmark(correlated, methods=["not-a-codec"])


def test_benchmark_rejects_unknown_transforms(correlated):
    with pytest.raises(ValueError, match="unknown transform"):
        egp.compress_benchmark(correlated, methods=["zlib-9"], transforms=("raw", "wavelet"))


def test_benchmark_lpc_transform_needs_an_order(correlated):
    with pytest.raises(ValueError, match="lpc_order"):
        egp.compress_benchmark(correlated, methods=["zlib-9"], transforms=("lpc",))


def test_no_codec_beats_the_entropy_rate_by_much(correlated):
    # Compressors cannot go below the entropy rate; allow a little slack for
    # the finite sample and the codecs' own modelling.
    best = min(r.bits_per_sample for r in egp.compress_benchmark(correlated, lpc_order=8))
    floor = egp.entropy_rate_approx(egp.from_preset("ma", taps=8).taps, 0.5)
    assert best > floor - 0.3


# --------------------------------------------------------------------------
# analytic approximation
# --------------------------------------------------------------------------


def test_innovation_variance_matches_the_factorized_leading_tap():
    # Independent of the cepstral construction: a spectral quadrature.
    for preset, kwargs in [("ma", dict(taps=8)), ("ar1", dict(rho=0.85, taps=64)),
                           ("lowpass", dict(cutoff=6000.0, fs=30000.0, taps=33))]:
        spec = egp.from_preset(preset, **kwargs)
        got = math.sqrt(egp.innovation_variance(spec.taps))
        assert got == pytest.approx(spec.h0, rel=2e-3), preset


def test_approximation_is_exact_for_white_noise():
    # For an i.i.d. process the entropy rate is the marginal entropy.
    for delta in (0.25, 1.0, 2.0):
        approx = egp.entropy_rate_approx(np.ones(1), delta)
        assert approx == pytest.approx(egp.marginal_entropy_bits(1.0, delta), abs=0.02)


def test_approximation_decomposes_into_floor_plus_spectral_term():
    spec = egp.from_preset("ar1", rho=0.9, taps=32)
    total = egp.entropy_rate_approx(spec.taps, 0.3)
    parts = egp.QUANTIZER_FLOOR + egp.snr_integral(spec.taps, 0.3)
    assert total == pytest.approx(parts, rel=1e-12)


def test_approximation_tracks_the_particle_filter():
    spec = egp.from_preset("ma", taps=8)
    res = egp.estimate_entropy_rate(spec, 0.5, n=20_000, n_particles=800, n_repeats=1, seed=0)
    assert not res.collapsed
    assert egp.entropy_rate_approx(spec.taps, 0.5) == pytest.approx(
        res.entropy_rate_bits, abs=0.1
    )
    # The estimate carries it for reference, so the two never disagree.
    assert res.approximation_bits == pytest.approx(egp.entropy_rate_approx(spec.taps, 0.5))


def test_collapsed_run_still_reports_the_approximation():
    # The reference value is most useful exactly when the filter has failed.
    spec = egp.from_preset("lowpass", cutoff=6000.0, fs=30000.0, taps=33)
    res = egp.estimate_entropy_rate(spec, 0.25, n=2_000, n_particles=300, n_repeats=1, seed=13)
    assert res.collapsed
    assert 0.0 < res.approximation_bits < res.upper_bound_bits


def test_cli_estimate_reports_the_approximation(capsys):
    assert main(["estimate", "--preset", "ma", "--taps", "4", "--delta", "0.5",
                 "-n", "2000", "-N", "200", "-r", "1", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "approx" in out and "vs estimate" in out


def test_lower_bound_is_below_the_approximation():
    for delta in (0.1, 0.5, 2.0):
        spec = egp.from_preset("ar1", rho=0.8, taps=32)
        assert egp.entropy_rate_lower_bound(spec.taps, delta) <= egp.entropy_rate_approx(
            spec.taps, delta
        )


def test_psd_grids_agree_away_from_nulls():
    spec = egp.from_preset("ar1", rho=0.7, taps=32)
    w_mid, s_mid = egp.psd(spec.taps, 256, midpoint=True)
    w_reg, s_reg = egp.psd(spec.taps, 256, midpoint=False)
    assert np.allclose(np.interp(w_mid, w_reg, s_reg), s_mid, rtol=0.02)


def test_fine_quantization_drives_the_approximation_to_the_classical_form():
    spec = egp.from_preset("ar1", rho=0.8, taps=64)
    delta = 0.01
    classical = egp.differential_entropy_bits(spec.h0) - math.log2(delta)
    assert egp.entropy_rate_approx(spec.taps, delta) == pytest.approx(classical, abs=0.01)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_approx_text_and_json(capsys):
    assert main(["approx", "--preset", "ma", "--taps", "8", "--delta", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "H' approx" in out and "quantizer floor" in out

    assert main(["approx", "--preset", "ma", "--taps", "8", "--delta", "0.5", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["units"] == "bits"
    assert data["entropy_rate_approx"] == pytest.approx(
        data["quantizer_floor"] + data["spectral_term"], rel=1e-12
    )


def test_cli_approx_nats_scale(capsys):
    main(["approx", "--preset", "ar1", "--rho", "0.8", "--delta", "0.5", "--json"])
    bits = json.loads(capsys.readouterr().out)
    main(["approx", "--preset", "ar1", "--rho", "0.8", "--delta", "0.5",
          "--units", "nats", "--json"])
    nats = json.loads(capsys.readouterr().out)
    assert nats["entropy_rate_approx"] == pytest.approx(
        bits["entropy_rate_approx"] * math.log(2.0)
    )


def test_cli_compress_json(capsys):
    rc = main(["compress", "--preset", "ma", "--taps", "4", "--delta", "0.5",
               "-n", "20000", "--methods", "zlib-9,lzma-9", "--seed", "0",
               "--json", "--quiet"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 20000 and out["baseline_bits"] == 16.0
    assert out["lpc_order"] == 3  # defaults to L - 1
    assert len(out["results"]) == 4
    assert out["ideal"]["entropy_rate"] is None
    assert math.isfinite(out["ideal"]["marginal_entropy"])


def test_cli_compress_with_estimate(capsys):
    rc = main(["compress", "--preset", "white", "--delta", "1.0", "-n", "20000",
               "--methods", "zlib-9", "--estimate", "--estimate-n", "20000",
               "--seed", "0", "--json", "--quiet"])
    assert rc == 0
    ideal = json.loads(capsys.readouterr().out)["ideal"]
    # for an i.i.d. process the entropy rate IS the marginal entropy
    assert abs(ideal["entropy_rate"] - ideal["marginal_entropy"]) < 0.02
    assert ideal["entropy_rate_collapsed"] is False


def test_cli_compress_table_and_progress(capsys):
    rc = main(["compress", "--preset", "white", "--delta", "1.0", "-n", "20000",
               "--methods", "zlib-9", "--lpc-order", "0", "--seed", "0"])
    assert rc == 0
    cap = capsys.readouterr()
    assert "baseline int16 = 16 bits/sample" in cap.out
    assert "H(Y_1) marginal bound" in cap.out
    assert "bits/sample" in cap.err and "generated 20000 samples" in cap.err
