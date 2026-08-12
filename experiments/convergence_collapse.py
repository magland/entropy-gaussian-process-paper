#!/usr/bin/env python
"""Section 5.1 experiment: the collapse diagnostic in the fine-quantization regime.

Runs the estimator on two strongly smoothing processes for which the state is
nearly determined by the observations, so that small particle clouds lose it
entirely: the 33-tap lowpass (cutoff 6 kHz at fs = 30 kHz) at delta = 0.25
sigma (delta/h0 about 6, state dimension 32), and the Gaussian smoothing
kernel with tau = 1.5 at delta = 0.5 sigma (delta/h0 about 8.6, state
dimension 12).  The point of the experiment is that the effective sample size
does not distinguish a collapsed run from a healthy one, while the per-step
log-likelihood diagnostic of Appendix B does.  For each N the results file
records the estimate, the per-replicate lost-lock fractions, the mean ESS,
and how many replicates were discarded.

Defaults are modest, sized for an integrated laptop GPU; raise --particles,
--samples and --repeats for the full-scale run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import PROCESS_BUILDERS, RESULTS_DIR, process_spec, run_case, run_meta, save_results

#: The quantization step used for each process unless --delta overrides it.
#: The gaussian15 step is 0.5 so that the collapse transition falls inside the
#: default particle grid; the others match the sigma/Delta = 4 regime of the
#: target application.
DEFAULT_DELTA = {"lowpass": 0.25, "gaussian15": 0.5, "gaussian1": 0.25, "ma": 0.25, "ar1": 0.25}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processes", nargs="+", default=["lowpass", "gaussian15"],
        choices=sorted(PROCESS_BUILDERS),
    )
    p.add_argument(
        "--particles", nargs="+", type=int, default=[300, 1000, 3000, 10_000, 30_000],
        help="grid of particle counts N",
    )
    p.add_argument("-n", "--samples", type=int, default=30_000, help="sequence length n")
    p.add_argument("-r", "--repeats", type=int, default=4)
    p.add_argument(
        "--delta", type=float, default=None,
        help="quantization step; the default is per-process (see DEFAULT_DELTA)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("-o", "--out", type=Path, default=RESULTS_DIR / "convergence_collapse.json")
    args = p.parse_args()

    runs = []
    for name in args.processes:
        spec = process_spec(name)
        delta = args.delta if args.delta is not None else DEFAULT_DELTA[name]
        for n_particles in sorted(args.particles):
            record = run_case(
                spec,
                delta,
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
