#!/usr/bin/env python
"""Targeted rerun of individual Section 5.2 points at a larger particle count.

The sweep of family_curves.py sets the particle count from h0 alone, which
occasionally leaves a point with a discarded replicate.  This script reruns
named points into a suffixed results file, which the figure and table
scripts merge over the main one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import RESULTS_DIR, run_meta, run_case, save_results
from family_curves import FAMILIES

#: (family, parameter value, delta, particles) for each point to redo.
POINTS = [
    ("lowpass", 0.16, 0.125, 260_000),
    ("lowpass", 0.25, 0.125, 130_000),
    ("lowpass", 0.25, 0.25, 130_000),
    ("lowpass", 0.25, 1.0, 130_000),
    ("lowpass", 0.25, 2.0, 130_000),
    ("lowpass", 0.3, 0.125, 130_000),
    ("lowpass", 0.3, 0.25, 130_000),
    ("lowpass", 0.3, 1.0, 130_000),
    ("lowpass", 0.3, 2.0, 130_000),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--samples", type=int, default=5000)
    p.add_argument("-r", "--repeats", type=int, default=16)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--engine", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("-o", "--out", type=Path, default=RESULTS_DIR / "family_curves_rerun.json")
    args = p.parse_args()

    panels: dict = {}
    for index, (name, value, delta, n_particles) in enumerate(POINTS):
        family = FAMILIES[name]
        spec = family["build"](value)
        record = run_case(
            spec,
            delta,
            n=args.samples,
            n_particles=n_particles,
            n_repeats=args.repeats,
            seed=args.seed + index,
            engine=args.engine,
            tag=f"{name} {family['param']}={value:g} (rerun)",
        )
        record["value"] = float(value)
        panel = panels.setdefault(name, {"kind": "param", "param": family["param"], "mc": []})
        panel["mc"].append(record)
        save_results(args.out, {"meta": run_meta(args.engine), "points": POINTS, "panels": panels})


if __name__ == "__main__":
    main()
