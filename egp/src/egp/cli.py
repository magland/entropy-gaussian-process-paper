"""Command line interface: ``egp presets | estimate | plot``."""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from . import spec as spec_mod
from .estimate import estimate_entropy_rate, marginal_entropy_bits
from .spec import PRESETS, ProcessSpec


def _load_array(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(float).ravel()
    text = open(path).read().replace(",", " ")
    return np.array([float(tok) for tok in text.split()], dtype=float)


def _add_process_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("process specification")
    source = group.add_mutually_exclusive_group()
    source.add_argument("--preset", choices=sorted(PRESETS), help="named process family")
    source.add_argument(
        "--kernel", metavar="FILE", help="FIR kernel applied to i.i.d. N(0,1) (.npy or text)"
    )
    source.add_argument(
        "--autocov", metavar="FILE", help="autocovariance at lags 0,1,... (.npy or text)"
    )
    group.add_argument("--taps", type=int, help="number of FIR taps (preset-dependent default)")
    group.add_argument(
        "--model-taps",
        type=int,
        help="length L of the minimum-phase model; defaults to the kernel length",
    )
    group.add_argument("--fs", type=float, default=1.0, help="sample rate in Hz (default: 1)")
    group.add_argument("--cutoff", type=float, help="cutoff in Hz for lowpass/highpass")
    group.add_argument(
        "--band", type=float, nargs=2, metavar=("LO", "HI"), help="band edges in Hz for bandpass"
    )
    group.add_argument("--rho", type=float, default=0.9, help="AR(1) correlation (default: 0.9)")
    group.add_argument("--tau", type=float, default=4.0, help="Gaussian kernel width in samples")
    group.add_argument("--window", default="hamming", help="FIR design window (default: hamming)")
    group.add_argument(
        "--sigma", type=float, default=1.0, help="process standard deviation (default: 1)"
    )
    group.add_argument(
        "--no-normalize",
        action="store_true",
        help="keep the kernel's natural scale instead of setting sigma",
    )
    group.add_argument(
        "--floor", type=float, default=1e-14, help="relative spectral floor"
    )
    group.add_argument("--nfft", type=int, help="FFT size for the spectral factorization")
    group.add_argument(
        "--delta", type=float, required=True, help="quantization step (rounding to multiples)"
    )


def build_spec(args: argparse.Namespace) -> ProcessSpec:
    sigma = None if args.no_normalize else args.sigma
    common = dict(sigma=sigma, fs=args.fs, floor=args.floor, nfft=args.nfft)
    if args.kernel:
        return spec_mod.from_taps(
            _load_array(args.kernel),
            n_taps=args.model_taps,
            label=f"kernel from {args.kernel}",
            **common,
        )
    if args.autocov:
        return spec_mod.from_autocov(
            _load_array(args.autocov),
            n_taps=args.model_taps,
            label=f"autocovariance from {args.autocov}",
            **common,
        )
    if not args.preset:
        raise SystemExit("error: specify one of --preset, --kernel, --autocov")
    return spec_mod.from_preset(
        args.preset,
        taps=args.taps,
        cutoff=args.cutoff,
        band=args.band,
        rho=args.rho,
        tau=args.tau,
        window=args.window,
        n_taps=args.model_taps,
        **common,
    )


def _describe(spec: ProcessSpec, delta: float) -> str:
    lines = [
        f"process     {spec.label}",
        f"taps L      {spec.n_taps}"
        f"    sigma {spec.sigma:.6g}    h0 {spec.h0:.6g}",
        f"delta       {delta:.6g}"
        f"    delta/sigma {delta / spec.sigma:.4g}    delta/h0 {delta / spec.h0:.4g}",
        f"factoring   autocov error {spec.autocov_error:.2e}"
        f"    floored bins {100 * spec.floored_fraction:.2f}%",
    ]
    if spec.floored_fraction > 0.01:
        lines.append(
            "  warning: the spectral floor is active on a large fraction of the grid, "
            "so h0 (and the entropy rate) partly reflects --floor rather than the process."
        )
    return "\n".join(lines)


def cmd_presets(args: argparse.Namespace) -> int:
    width = max(len(name) for name in PRESETS)
    for name in sorted(PRESETS):
        print(f"  {name:<{width}}  {PRESETS[name]}")
    return 0


def _engine_desc(engine: str) -> str:
    if engine != "gpu":
        return engine
    try:
        from .pf_gpu import gpu_description

        return f"gpu ({gpu_description()})"
    except Exception:
        return "gpu"


def cmd_estimate(args: argparse.Namespace) -> int:
    spec = build_spec(args)
    if not args.quiet:
        print(_describe(spec, args.delta))
        print(
            f"filter      n {args.n}    particles {args.particles}    repeats {args.repeats}"
            f"    engine {_engine_desc(args.engine)}",
            flush=True,
        )

    width = 46
    # The in-place counter only means anything on a terminal; piped or
    # redirected it would bury the per-repeat lines under thousands of rewrites.
    tty = sys.stderr.isatty()

    def progress(rep, n_reps, done, total):
        if not tty:
            return
        pct = 100.0 * done / total
        sys.stderr.write(f"\r  [{rep + 1}/{n_reps}] {pct:5.1f}%".ljust(width))
        sys.stderr.flush()

    def on_repeat(u):
        se = "     --" if np.isnan(u.running_stderr_bits) else f"{u.running_stderr_bits:7.4f}"
        mean = "      --" if np.isnan(u.running_mean_bits) else f"{u.running_mean_bits:8.4f}"
        flag = "" if u.usable else "  LOST LOCK - discarded"
        if tty:
            sys.stderr.write("\r" + " " * width + "\r")
        sys.stderr.write(
            f"  [{u.index + 1}/{u.n_repeats}] {u.entropy_rate_bits:8.4g} bits/sample   "
            f"mean {mean} +/- {se}   "
            f"ESS {100 * u.mean_ess:4.1f}%   ({u.seconds:5.1f}s){flag}\n"
        )
        sys.stderr.flush()

    result = estimate_entropy_rate(
        spec,
        args.delta,
        n=args.n,
        n_particles=args.particles,
        n_repeats=args.repeats,
        burn_in=args.burn_in,
        seed=args.seed,
        progress=None if args.quiet else progress,
        on_repeat=None if args.quiet else on_repeat,
        engine=args.engine,
    )
    if not args.quiet and tty:
        sys.stderr.write("\r" + " " * width + "\r")
        sys.stderr.flush()

    print()
    if result.collapsed:
        print("FILTER COLLAPSED - no estimate reported")
        print(
            f"  the particle cloud lost the state on "
            f"{100 * result.lost_lock_fraction:.1f}% of steps"
            + ("" if result.within_bounds else ", and the result violates a rigorous bound")
        )
        print(f"  (the unusable value was {result.entropy_rate_bits:.4g} bits/sample)")
        print(f"  for reference the analytic approximation is "
              f"{result.approximation_bits:.4f} bits/sample")
        print(
            f"  increase --particles well beyond {result.n_particles}; this process needs "
            f"many particles because delta/h0 = {result.delta_over_h0:.3g} with L = {result.n_taps}"
        )
    else:
        print(
            f"entropy rate   {result.entropy_rate_bits:.4f} +/- "
            f"{result.stderr_bits:.4f} bits/sample"
        )
        if result.n_failed_repeats:
            print(
                f"  WARNING      {result.n_failed_repeats} of {result.n_repeats} replicates lost "
                f"the state and were discarded; the value above averages the "
                f"{result.n_usable_repeats} that survived."
            )
            print(
                f"               treat it as provisional - the same --particles "
                f"{result.n_particles} produced the failures. Increase it."
            )
        print(
            f"  bounds       {result.lower_bound_bits:.4f}  <=  est  <=  "
            f"{result.upper_bound_bits:.4f}"
        )
        print(
            f"  approx       {result.approximation_bits:.4f} bits/sample  "
            f"({result.approximation_bits - result.entropy_rate_bits:+.4f} vs estimate)"
        )
        print(f"  order-0      {result.empirical_symbol_entropy_bits:.4f} bits/sample (memoryless)")
        print(
            "  per repeat   "
            + ", ".join(
                f"{v:.4f}" if ok else f"{v:.3g} (discarded)"
                for v, ok in zip(result.per_repeat_bits, result.per_repeat_usable)
            )
        )
    print(
        f"  ESS          mean {100 * result.mean_ess:.1f}% of N, "
        f"min {100 * result.min_ess:.1f}%"
    )
    print(
        f"  symbols      {result.n_symbols_observed}    burn-in {result.burn_in}"
        f"    elapsed {result.elapsed_sec:.1f} s"
    )
    if not result.collapsed and result.mean_ess < 0.05:
        print("  warning: low effective sample size; increase --particles")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)
        print(f"  wrote {args.json}")
    return 1 if result.collapsed else 0


def cmd_approx(args: argparse.Namespace) -> int:
    from .approx import (
        QUANTIZER_FLOOR,
        entropy_rate_approx,
        entropy_rate_lower_bound,
        innovation_variance,
        snr_integral,
    )

    spec = build_spec(args)
    delta = args.delta
    scale = 1.0 if args.units == "bits" else math.log(2.0)

    approx = entropy_rate_approx(spec.taps, delta, args.grid)
    spectral = snr_integral(spec.taps, delta, args.grid)
    lower = entropy_rate_lower_bound(spec.taps, delta)
    variance = innovation_variance(spec.taps)
    classical = (
        0.5 * math.log2(2 * math.pi * math.e * variance) - math.log2(delta)
        if variance > 0
        else -math.inf
    )
    marginal = marginal_entropy_bits(spec.sigma, delta)

    if args.json:
        print(json.dumps({
            "process": spec.label,
            "entropy_rate_approx": approx * scale,
            "quantizer_floor": QUANTIZER_FLOOR * scale,
            "spectral_term": spectral * scale,
            "lower_bound": lower * scale,
            "classical": classical * scale if math.isfinite(classical) else None,
            "marginal_entropy": marginal * scale,
            "units": args.units,
            "L": spec.n_taps,
            "delta": delta,
            "sigma": spec.sigma,
            "h0": spec.h0,
            "innovation": math.sqrt(variance),
            "grid": args.grid,
        }))
        return 0

    print(_describe(spec, delta))
    print()
    print(f"H' approx = {approx * scale:.5f} {args.units}/sample")
    print(f"          = {QUANTIZER_FLOOR * scale:.5f} (quantizer floor) "
          f"+ {spectral * scale:.5f} (spectral SNR integral)")
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    from .approx import entropy_rate_approx
    from .compression import compress_benchmark, empirical_entropy_bits, quantized_sample

    spec = build_spec(args)
    delta = args.delta
    order = spec.n_taps - 1 if args.lpc_order is None else args.lpc_order
    methods = args.methods.split(",") if args.methods else None
    transforms = args.transforms.split(",") if args.transforms else None

    y = quantized_sample(spec.taps, delta, args.samples, np.random.default_rng(args.seed))
    distinct = int(np.unique(y).size)
    if not args.quiet:
        print(f"generated {y.size} samples, {distinct} distinct values, "
              f"range [{int(y.min())}, {int(y.max())}]", file=sys.stderr, flush=True)

    def report(event, info):
        if args.quiet or event != "codec":
            return
        r = info["result"]
        status = r.error or f"{r.bits_per_sample:7.3f} bits/sample   {r.ratio:5.2f}x"
        print(f"  {r.method:>10} {r.transform:>10}   {status}   ({r.seconds:5.1f}s)",
              file=sys.stderr, flush=True)

    results = compress_benchmark(
        y, lpc_order=order, transforms=transforms, methods=methods, progress=report
    )

    estimate = None
    if args.estimate:
        estimate = estimate_entropy_rate(
            spec, delta, n=args.estimate_n, n_particles=args.particles,
            n_repeats=args.repeats, seed=args.seed, engine=args.engine,
        )

    ideal = {
        "marginal_entropy": marginal_entropy_bits(spec.sigma, delta),
        "empirical_entropy": empirical_entropy_bits(y),
        "approximation": entropy_rate_approx(spec.taps, delta),
        "entropy_rate": estimate.entropy_rate_bits if estimate else None,
        "entropy_rate_stderr": estimate.stderr_bits if estimate else None,
        "entropy_rate_collapsed": estimate.collapsed if estimate else None,
    }
    if args.json:
        print(json.dumps({
            "process": spec.label,
            "n": int(y.size),
            "distinct": distinct,
            "delta": delta,
            "sigma": spec.sigma,
            "h0": spec.h0,
            "L": spec.n_taps,
            "lpc_order": order,
            "baseline_bits": 16.0,
            "ideal": ideal,
            "results": [vars(r) for r in results],
        }))
        return 0

    print(_describe(spec, delta))
    print(f"n={y.size} samples   {distinct} distinct values   "
          f"baseline int16 = 16 bits/sample")
    print()
    print(f"{'method':<11} {'transform':<11} {'bytes':>10} {'bits/sample':>12} {'ratio':>8}")
    for r in results:
        if r.error:
            print(f"{r.method:<11} {r.transform:<11} {'failed: ' + r.error[:40]:>32}")
        else:
            print(f"{r.method:<11} {r.transform:<11} {r.n_bytes:>10} "
                  f"{r.bits_per_sample:>12.3f} {r.ratio:>7.2f}x")
    print()
    print(f"{'ideal':<11} {'H(Y_1) marginal bound':<23} {ideal['marginal_entropy']:>12.3f}")
    print(f"{'':<11} {'empirical order-0':<23} {ideal['empirical_entropy']:>12.3f}")
    print(f"{'':<11} {'analytic approximation':<23} {ideal['approximation']:>12.3f}")
    if estimate is not None:
        if estimate.collapsed:
            print(f"{'':<11} {'particle filter':<23} {'COLLAPSED':>12}")
        else:
            print(f"{'':<11} {'particle filter':<23} "
                  f"{estimate.entropy_rate_bits:>12.3f}  +/- {estimate.stderr_bits:.3f}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    from .plotting import make_figure

    spec = build_spec(args)
    print(_describe(spec, args.delta))
    fig = make_figure(
        spec, args.delta, n=args.n, n_show=args.n_show, seed=args.seed, nfft=args.nfft_plot
    )
    if args.out:
        fig.savefig(args.out, dpi=args.dpi)
        print(f"wrote {args.out}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="egp",
        description="Entropy rate of a quantized stationary Gaussian process.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_presets = sub.add_parser("presets", help="list the built-in process presets")
    p_presets.set_defaults(func=cmd_presets)

    p_est = sub.add_parser(
        "estimate",
        help="estimate the entropy rate with a particle filter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_process_args(p_est)
    run = p_est.add_argument_group("estimation")
    run.add_argument("-n", "--n", type=int, default=50_000, help="sequence length")
    run.add_argument("-N", "--particles", type=int, default=1_000, help="number of particles")
    run.add_argument("-r", "--repeats", type=int, default=3, help="independent replicates")
    run.add_argument(
        "--burn-in",
        type=int,
        default=None,
        help="leading steps filtered but excluded from the rate",
    )
    run.add_argument("--seed", type=int, default=0, help="random seed")
    run.add_argument(
        "--engine",
        choices=["cpu", "gpu"],
        default="cpu",
        help="particle filter engine; gpu runs the WGSL shader via wgpu (pip install egp[gpu])",
    )
    run.add_argument("--json", metavar="FILE", help="also write the result as JSON")
    run.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    p_est.set_defaults(func=cmd_estimate)

    p_apx = sub.add_parser(
        "approx",
        help="analytic approximation to the entropy rate from the spectrum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_process_args(p_apx)
    apx = p_apx.add_argument_group("approximation")
    apx.add_argument("--units", choices=["bits", "nats"], default="bits")
    apx.add_argument("--grid", type=int, default=16384, help="spectral quadrature points")
    apx.add_argument("--json", action="store_true", help="machine-readable output")
    p_apx.set_defaults(func=cmd_approx)

    p_comp = sub.add_parser(
        "compress",
        help="bits per sample achieved by real compressors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_process_args(p_comp)
    comp = p_comp.add_argument_group("compression")
    comp.add_argument("-n", "--samples", type=int, default=1_000_000, help="time series length")
    comp.add_argument(
        "--lpc-order", type=int, help="linear prediction order (default: L-1; 0 disables)"
    )
    comp.add_argument("--methods", help="comma-separated subset of the available codecs")
    comp.add_argument(
        "--transforms",
        help="comma-separated stages to code: raw, delta, lpc (default: raw,lpc)",
    )
    comp.add_argument(
        "--estimate", action="store_true", help="also run the particle filter for the ideal"
    )
    comp.add_argument(
        "--estimate-n", type=int, default=50_000, help="sequence length for --estimate"
    )
    comp.add_argument("-N", "--particles", type=int, default=1_000, help="particles for --estimate")
    comp.add_argument("-r", "--repeats", type=int, default=1, help="replicates for --estimate")
    comp.add_argument(
        "--engine", choices=["cpu", "gpu"], default="cpu", help="filter engine for --estimate"
    )
    comp.add_argument("--seed", type=int, default=0, help="random seed")
    comp.add_argument("--json", action="store_true", help="machine-readable output")
    comp.add_argument("-q", "--quiet", action="store_true", help="suppress progress on stderr")
    p_comp.set_defaults(func=cmd_compress)

    p_plot = sub.add_parser(
        "plot",
        help="figure showing the properties of a process (no estimation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_process_args(p_plot)
    fig = p_plot.add_argument_group("figure")
    fig.add_argument("-o", "--out", default="egp.png", help="output image path")
    fig.add_argument("--n", type=int, default=200_000, help="samples drawn for the statistics")
    fig.add_argument("--n-show", type=int, default=400, help="samples drawn in the trace panel")
    fig.add_argument("--nfft-plot", type=int, default=8192, help="FFT size for the spectrum panel")
    fig.add_argument("--dpi", type=int, default=130, help="output resolution")
    fig.add_argument("--seed", type=int, default=0, help="random seed")
    fig.add_argument("--show", action="store_true", help="open an interactive window")
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
