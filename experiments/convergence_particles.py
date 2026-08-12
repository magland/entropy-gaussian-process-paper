#!/usr/bin/env python
"""Section 5.1 experiment: convergence of the estimator in the particle count N.

Runs the estimator at an increasing sequence of particle counts on the same
observed sequences: the seed is fixed, so each replicate sees an identical
symbol sequence at every N and only the filter's internal randomness differs.
The paired difference against the largest N therefore isolates the filter's
O(1/N) upward bias from the sequence-to-sequence variation, which is common
to all N.  The default processes span the range of bias sizes the filter
exhibits at delta = sigma/4, from the first difference (under a millibit at
N = 250) to the wide moving averages (hundreds of millibits).

Defaults are modest, sized for an integrated laptop GPU; raise --particles,
--samples and --repeats for the full-scale run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import PROCESS_BUILDERS, RESULTS_DIR, process_spec, run_case, run_meta, save_results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processes", nargs="+",
        default=["diff", "gaussian1", "gaussian125", "ma", "ma16", "ma32", "ma64"],
        choices=sorted(PROCESS_BUILDERS),
    )
    p.add_argument(
        "--particles", nargs="+", type=int, default=[125, 500, 2000, 8000, 32000],
        help="grid of particle counts N",
    )
    p.add_argument("-n", "--samples", type=int, default=20_000, help="sequence length n")
    p.add_argument("-r", "--repeats", type=int, default=8)
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("-o", "--out", type=Path, default=RESULTS_DIR / "convergence_particles.json")
    args = p.parse_args()

    runs = []
    for name in args.processes:
        spec = process_spec(name)
        for n_particles in sorted(args.particles):
            record = run_case(
                spec,
                args.delta,
                n=args.samples,
                n_particles=n_particles,
                n_repeats=args.repeats,
                seed=args.seed,
                engine=args.engine,
                tag=f"{name}: {spec.label}",
            )
            record["process"] = name
            runs.append(record)

    config = {
        "processes": args.processes,
        "particles": sorted(args.particles),
        "samples": args.samples,
        "repeats": args.repeats,
        "delta": args.delta,
        "seed": args.seed,
    }
    save_results(args.out, {"meta": run_meta(args.engine), "config": config, "runs": runs})


if __name__ == "__main__":
    main()
