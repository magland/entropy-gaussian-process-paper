#!/usr/bin/env python
"""Section 5.1 experiment: convergence of the estimator in the sequence length n.

Runs the estimator at a fixed, comfortably large particle count over an
increasing sequence of lengths n, with the number of replicates held fixed, so
that the plot shows both the stability of the mean and the 1/sqrt(n) shrinkage
of the per-replicate scatter.  The seed is fixed, so shorter sequences are
prefixes of longer ones drawn from the same generator state.

Defaults are modest, sized for an integrated laptop GPU; raise --lengths,
--particles and --repeats for the full-scale run.
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
        "--lengths", nargs="+", type=int, default=[2500, 5000, 10_000, 20_000, 40_000],
        help="grid of sequence lengths n",
    )
    p.add_argument("-N", "--particles", type=int, default=4000)
    p.add_argument(
        "--hard-particles", type=int, default=None,
        help="particle count for processes with h0 < 0.2 (the wide moving "
             "averages), which need more particles to keep lock at the longest "
             "lengths; default 4x --particles",
    )
    p.add_argument("-r", "--repeats", type=int, default=8)
    p.add_argument(
        "--hard-repeats", type=int, default=None,
        help="replicates for the h0 < 0.2 processes (default: --repeats)",
    )
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("-o", "--out", type=Path, default=RESULTS_DIR / "convergence_length.json")
    args = p.parse_args()

    hard_particles = args.hard_particles or 4 * args.particles
    runs = []
    for name in args.processes:
        spec = process_spec(name)
        hard = spec.h0 < 0.2
        for n in sorted(args.lengths):
            record = run_case(
                spec,
                args.delta,
                n=n,
                n_particles=hard_particles if hard else args.particles,
                n_repeats=(args.hard_repeats or args.repeats) if hard else args.repeats,
                seed=args.seed,
                engine=args.engine,
                tag=f"{name}: {spec.label}",
            )
            record["process"] = name
            runs.append(record)

    config = {
        "processes": args.processes,
        "lengths": sorted(args.lengths),
        "particles": args.particles,
        "hard_particles": hard_particles,
        "repeats": args.repeats,
        "hard_repeats": args.hard_repeats or args.repeats,
        "delta": args.delta,
        "seed": args.seed,
    }
    save_results(args.out, {"meta": run_meta(args.engine), "config": config, "runs": runs})


if __name__ == "__main__":
    main()
