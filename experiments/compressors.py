#!/usr/bin/env python
"""Section 5.3 experiment: practical lossless coders against the entropy rate.

Generates long quantized realizations of each family and compresses them with
two representative coders: zlib (level 9) on the raw int16 symbol stream, and
the package's fixed-point linear predictor (FLAC style, order 32 for every
family, the maximum order of FLAC) followed by an ANS entropy coder on the
residuals.  The uniform order matters: the natural-seeming choice L - 1 is
far worse for the families with spectral zeros (moving average, first
difference), whose AR representations decay slowly.  Both stages of both coders are
recorded (raw and LPC residual), together with the analytical approximation
of the entropy rate, which Section 5.2 validated in this range of steps, and
the empirical order-0 symbol entropy for context.  For white noise the
predictor has order 0 and the residual stage coincides with the raw stream.

The strongly smoothing families are restricted to delta >= 0.25 sigma, the
range in which the reference rate was validated directly.  Everything is on
the CPU; the whole run takes a few minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import egp
from egp.compression import compress_benchmark, quantized_sample
from common import PROCESS_BUILDERS, RESULTS_DIR, process_spec, run_meta, save_results

HARD = ("gaussian15", "lowpass")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processes", nargs="+",
        default=["white", "diff", "ma", "ar1", "gaussian1", "gaussian15", "lowpass"],
        choices=sorted(PROCESS_BUILDERS),
    )
    p.add_argument("--deltas", nargs="+", type=float, default=[0.0625, 0.125, 0.25, 0.5, 1.0])
    p.add_argument("--hard-min-delta", type=float, default=0.25)
    p.add_argument("-n", "--samples", type=int, default=200_000)
    p.add_argument("-r", "--repeats", type=int, default=3)
    p.add_argument("--lpc-order", type=int, default=32)
    p.add_argument("--methods", nargs="+", default=["zlib-9", "ans"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", type=Path, default=RESULTS_DIR / "compressors.json")
    args = p.parse_args()

    runs = []
    for name in args.processes:
        spec = process_spec(name)
        lpc_order = args.lpc_order
        for delta in sorted(args.deltas):
            if name in HARD and delta < args.hard_min_delta:
                print(f"--- {name}: SKIPPED delta={delta:g} (below --hard-min-delta)", flush=True)
                continue
            codecs: dict[str, list[float]] = {}
            order0 = []
            for child in np.random.SeedSequence(args.seed).spawn(args.repeats):
                y = quantized_sample(spec.taps, delta, args.samples, np.random.default_rng(child))
                order0.append(egp.empirical_symbol_entropy_bits(y))
                for res in compress_benchmark(y, lpc_order=lpc_order, methods=args.methods):
                    if res.error:
                        raise RuntimeError(f"{name} delta={delta}: {res.method} failed: {res.error}")
                    codecs.setdefault(f"{res.method}/{res.transform}", []).append(
                        res.bits_per_sample
                    )
            record = {
                "process": name,
                "delta": delta,
                "approx_bits": egp.entropy_rate_approx(spec.taps, delta),
                "order0_bits": float(np.mean(order0)),
                "lpc_order": lpc_order,
                "codecs": codecs,
            }
            runs.append(record)
            summary = "  ".join(
                f"{k} {np.mean(v):.4f}" for k, v in sorted(codecs.items())
            )
            print(
                f"--- {name}  delta={delta:g}  approx {record['approx_bits']:.4f}  {summary}",
                flush=True,
            )

    config = {
        "processes": args.processes,
        "deltas": sorted(args.deltas),
        "hard_min_delta": args.hard_min_delta,
        "samples": args.samples,
        "repeats": args.repeats,
        "lpc_order": args.lpc_order,
        "methods": args.methods,
        "seed": args.seed,
    }
    save_results(args.out, {"meta": run_meta("cpu"), "config": config, "runs": runs})


if __name__ == "__main__":
    main()
