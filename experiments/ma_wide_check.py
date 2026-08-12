#!/usr/bin/env python
"""Is the wide-moving-average discrepancy of Section 5.2 filter bias?

At the widest moving averages and the finest steps the particle-filter
estimate sits *above* the analytical approximation, which the analysis of
Section 3.4 says should not happen once the filter has converged, since both
methods are biased upward.  Sweeping the particle count at a fixed point
distinguishes the possibilities: residual filter bias falls toward zero as N
grows, whereas a genuine error of the approximation does not.

The moving average is the family the convergence study of Section 5.1 found
hardest for the filter, its spectrum vanishing at each null of the boxcar
response, so slow convergence here is expected; this script measures how
slow.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import egp
from common import RESULTS_DIR, run_meta, save_results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=64, help="moving-average width")
    p.add_argument("--delta", type=float, default=0.125)
    p.add_argument("--particles", nargs="+", type=int, default=[30_000, 130_000, 400_000])
    p.add_argument("-n", "--samples", type=int, default=5000)
    p.add_argument("-r", "--repeats", type=int, default=16)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("-o", "--out", type=Path, default=RESULTS_DIR / "ma_wide_check.json")
    args = p.parse_args()

    spec = egp.from_preset("ma", taps=args.width)
    approx = egp.entropy_rate_approx(spec.taps, args.delta)
    print(f"MA w={args.width}  h0={spec.h0:.4f}  approx={approx:.4f}", flush=True)

    runs = []
    for n_particles in args.particles:
        started = time.time()
        result = egp.estimate_entropy_rate(
            spec,
            args.delta,
            n=args.samples,
            n_particles=n_particles,
            n_repeats=args.repeats,
            seed=args.seed,
            engine=args.engine,
        )
        error_mb = 1000.0 * (result.approximation_bits - result.entropy_rate_bits)
        print(
            f"N={n_particles:7d}: {result.entropy_rate_bits:.4f} +/- {result.stderr_bits:.4f}"
            f"   error {error_mb:+.1f} mb   failed {result.n_failed_repeats}/{args.repeats}"
            f"   ({time.time() - started:.0f}s)",
            flush=True,
        )
        runs.append({"result": result.to_dict(), "error_millibits": error_mb})
        save_results(
            args.out,
            {
                "meta": run_meta(args.engine),
                "config": {
                    "width": args.width,
                    "delta": args.delta,
                    "particles": args.particles,
                    "samples": args.samples,
                    "repeats": args.repeats,
                    "seed": args.seed,
                },
                "runs": runs,
            },
        )


if __name__ == "__main__":
    main()
