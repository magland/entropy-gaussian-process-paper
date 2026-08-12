#!/usr/bin/env python
"""Section 5.2 experiment: the approximation against the estimator, per family.

For each parameterized process family, the analytical approximation of
Section 3 is evaluated on a dense parameter grid at each quantization step,
and the particle-filter estimator is run at a subset of the grid points
(about eight per curve), so that the two may be compared at every point of
the sweep.  The families are

  ma        moving average of width w (w = 1 is white noise)
  ar1       AR(1) with correlation rho, truncated to 64 taps
  gaussian  Gaussian kernel of width tau, 2*ceil(4 tau)+1 taps
  lowpass   windowed-sinc lowpass, 33 taps, as a function of cutoff / fs

White noise and the first difference have no family parameter and are
instead swept in the quantization step itself: the approximation on a dense
grid of Delta/sigma and the estimator at a log-spaced subset, out to the
coarse regime where the approximation fails.

The particle count of each point is set from its one-step prediction error
h0, which is what governs how hard the filtering problem is: points with
h0 above --hard-h0 use --particles, those below use --hard-particles, and
those below --very-hard-h0 use --very-hard-particles.  Results are written
after every point, so a run may be interrupted without losing what it has
computed, and the figure and table scripts merge several results files, so
any point that loses lock may be rerun on its own at a larger particle
count.  Points whose replicates all lose lock are recorded as such and
appear as marked gaps in the figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import egp
from common import RESULTS_DIR, run_case, run_meta, save_results

DELTAS = [0.125, 0.25, 1.0, 2.0]

#: The Delta/sigma sweep for the two non-parameterized processes.
DELTA_SWEEP_MC = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
DELTA_SWEEP_DENSE = np.geomspace(0.1, 4.5, 40)


def gaussian_taps(tau: float) -> int:
    return 2 * int(np.ceil(4.0 * tau)) + 1


FAMILIES = {
    "ma": {
        "kind": "param",
        "param": "width",
        "build": lambda w: egp.from_preset("ma", taps=int(round(w))),
        "dense": sorted(set(int(round(v)) for v in np.geomspace(1, 64, 25))),
        "mc": [1, 2, 4, 6, 8, 16, 32, 64],
    },
    "ar1": {
        "kind": "param",
        "param": "rho",
        "build": lambda rho: egp.from_preset("ar1", rho=float(rho), taps=64),
        "dense": np.linspace(0.0, 0.95, 20),
        "mc": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95],
    },
    "gaussian": {
        "kind": "param",
        "param": "tau",
        "build": lambda tau: egp.from_preset(
            "gaussian", tau=float(tau), taps=gaussian_taps(float(tau))
        ),
        "dense": np.linspace(0.25, 1.5, 21),
        "mc": [0.25, 0.45, 0.65, 0.85, 1.0, 1.15, 1.3, 1.5],
    },
    "lowpass": {
        "kind": "param",
        "param": "cutoff",
        "build": lambda c: egp.from_preset("lowpass", cutoff=float(c), fs=1.0, taps=33),
        "dense": np.linspace(0.08, 0.45, 25),
        "mc": [0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.36, 0.45],
    },
    "white": {
        "kind": "delta",
        "build": lambda: egp.from_preset("white"),
    },
    "diff": {
        "kind": "delta",
        "build": lambda: egp.from_preset("diff"),
    },
}


def dense_param_curves(family: dict, deltas: list[float]) -> dict:
    values, rates, h0s, floored = [], [], [], []
    for value in family["dense"]:
        spec = family["build"](value)
        values.append(float(value))
        h0s.append(spec.h0)
        floored.append(spec.floored_fraction)
        rates.append([egp.entropy_rate_approx(spec.taps, d) for d in deltas])
    return {"values": values, "rates": rates, "h0": h0s, "floored_fraction": floored}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    p.add_argument("--deltas", nargs="+", type=float, default=DELTAS)
    p.add_argument("-N", "--particles", type=int, default=1000)
    p.add_argument("--hard-particles", type=int, default=4000)
    p.add_argument("--very-hard-particles", type=int, default=16000)
    p.add_argument(
        "--hard-h0", type=float, default=0.3,
        help="points with h0 below this use the hard tier",
    )
    p.add_argument(
        "--very-hard-h0", type=float, default=0.05,
        help="points with h0 below this use the very-hard tier",
    )
    p.add_argument("-n", "--samples", type=int, default=8000)
    p.add_argument("-r", "--repeats", type=int, default=3)
    p.add_argument(
        "--hard-samples", type=int, default=None,
        help="sequence length for the hard tiers (default: --samples). Since every "
             "step is an opportunity to lose lock, the hard points are cheaper to "
             "run as shorter sequences with more replicates at the same precision.",
    )
    p.add_argument("--hard-repeats", type=int, default=None)
    p.add_argument("--very-hard-samples", type=int, default=None)
    p.add_argument("--very-hard-repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("-o", "--out", type=Path, default=RESULTS_DIR / "family_curves.json")
    args = p.parse_args()

    deltas = sorted(args.deltas)
    panels = {}
    case_index = 0

    config = {
        "families": args.families,
        "deltas": deltas,
        "delta_sweep_mc": DELTA_SWEEP_MC,
        "particles": args.particles,
        "hard_particles": args.hard_particles,
        "very_hard_particles": args.very_hard_particles,
        "hard_h0": args.hard_h0,
        "very_hard_h0": args.very_hard_h0,
        "samples": args.samples,
        "repeats": args.repeats,
        "hard_samples": args.hard_samples,
        "hard_repeats": args.hard_repeats,
        "very_hard_samples": args.very_hard_samples,
        "very_hard_repeats": args.very_hard_repeats,
        "seed": args.seed,
    }

    def checkpoint() -> None:
        """Write what has been computed so far, so a run may be interrupted."""
        save_results(
            args.out,
            {
                "meta": run_meta(args.engine),
                "config": config,
                "runs_kind": "family_curves",
                "panels": panels,
            },
        )

    def tier_for(spec) -> tuple[int, int, int]:
        """(particles, sequence length, replicates) for this point's difficulty."""
        if spec.h0 < args.very_hard_h0:
            return (
                args.very_hard_particles,
                args.very_hard_samples or args.samples,
                args.very_hard_repeats or args.repeats,
            )
        if spec.h0 < args.hard_h0:
            return (
                args.hard_particles,
                args.hard_samples or args.samples,
                args.hard_repeats or args.repeats,
            )
        return args.particles, args.samples, args.repeats

    def mc_case(spec, delta, tag):
        nonlocal case_index
        n_particles, n, n_repeats = tier_for(spec)
        record = run_case(
            spec,
            delta,
            n=n,
            n_particles=n_particles,
            n_repeats=n_repeats,
            seed=args.seed + case_index,
            engine=args.engine,
            tag=tag,
        )
        case_index += 1
        return record

    for name in args.families:
        family = FAMILIES[name]
        if family["kind"] == "param":
            mc_runs = []
            panels[name] = {
                "kind": "param",
                "param": family["param"],
                "deltas": deltas,
                "dense": dense_param_curves(family, deltas),
                "mc": mc_runs,
            }
            for value in family["mc"]:
                spec = family["build"](value)
                for delta in deltas:
                    record = mc_case(spec, delta, f"{name} {family['param']}={value:g}")
                    record["value"] = float(value)
                    mc_runs.append(record)
                    checkpoint()
        else:
            spec = family["build"]()
            mc_runs = []
            panels[name] = {
                "kind": "delta",
                "dense": {
                    "values": [float(d) for d in DELTA_SWEEP_DENSE],
                    "rates": [
                        egp.entropy_rate_approx(spec.taps, d) for d in DELTA_SWEEP_DENSE
                    ],
                    "h0": spec.h0,
                },
                "mc": mc_runs,
            }
            for delta in DELTA_SWEEP_MC:
                record = mc_case(spec, delta, f"{name} delta={delta:g}")
                record["value"] = float(delta)
                mc_runs.append(record)
                checkpoint()

    checkpoint()


if __name__ == "__main__":
    main()
