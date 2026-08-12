#!/usr/bin/env python
"""Section 5.3 experiment: practical lossless coders against the entropy rate.

Generates long quantized realizations of each family and compresses them with
a panel of coders.  Five are general-purpose byte compressors (zlib, bzip2,
and LZMA from the standard library, zstd, and brotli, each at its highest
level); one is the reference FLAC encoder, the standard for lossless audio,
which does its own linear prediction and Rice coding; and one is an ANS
entropy coder, which has no model of its own and therefore measures whatever
structure the preceding transform has already removed.

Every codec is run on three integer-reversible views of the same stream: the
raw symbols, their first difference, and the residual of the package's
fixed-point linear predictor (FLAC style, order 32 for every family, the
maximum order of FLAC).  The uniform order matters: the natural-seeming
choice L - 1 is far worse for the families with spectral zeros (moving
average, first difference), whose AR representations decay slowly.  Each
record also carries the analytical approximation of the entropy rate, which
Section 5.2 validated in this range of steps, and the empirical order-0
symbol entropy for context.

The strongly smoothing families are restricted to delta >= 0.25 sigma, the
range in which the reference rate was validated directly.  Everything is on
the CPU; the default panel over the default grid takes about an hour, most of
it in brotli.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import egp
from egp.compression import compress_benchmark, quantized_sample
from common import PROCESS_BUILDERS, RESULTS_DIR, process_spec, run_meta, save_results

HARD = ("gaussian15", "lowpass")


def run_point(name: str, delta: float, args: argparse.Namespace) -> dict:
    """One grid point: compress every replicate with every codec.

    Each point draws its replicates from the same seed sequence, so the result
    does not depend on how the points are distributed over workers.
    """
    spec = process_spec(name)
    codecs: dict[str, list[float]] = {}
    order0 = []
    for child in np.random.SeedSequence(args.seed).spawn(args.repeats):
        y = quantized_sample(spec.taps, delta, args.samples, np.random.default_rng(child))
        order0.append(egp.empirical_symbol_entropy_bits(y))
        for res in compress_benchmark(
            y, lpc_order=args.lpc_order, transforms=args.transforms, methods=args.methods
        ):
            if res.error:
                raise RuntimeError(f"{name} delta={delta}: {res.method} failed: {res.error}")
            codecs.setdefault(f"{res.method}/{res.transform}", []).append(res.bits_per_sample)
    return {
        "process": name,
        "delta": delta,
        "approx_bits": egp.entropy_rate_approx(spec.taps, delta),
        "order0_bits": float(np.mean(order0)),
        "lpc_order": args.lpc_order,
        "codecs": codecs,
    }


def report(record: dict) -> None:
    """One line per grid point: the best codec on each transform."""
    best: dict[str, tuple[str, float]] = {}
    for key, values in record["codecs"].items():
        method, transform = key.split("/", 1)
        bits = float(np.mean(values))
        if transform not in best or bits < best[transform][1]:
            best[transform] = (method, bits)
    summary = "  ".join(
        f"{transform}: {method} {bits:.4f}" for transform, (method, bits) in sorted(best.items())
    )
    print(
        f"--- {record['process']}  delta={record['delta']:g}  "
        f"approx {record['approx_bits']:.4f}  {summary}",
        flush=True,
    )


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
    p.add_argument(
        "--methods", nargs="+",
        default=["zlib-9", "bz2-9", "lzma-9", "zstd-19", "brotli-11", "flac-8", "ans"],
    )
    p.add_argument(
        "--transforms", nargs="+", default=["raw", "delta", "lpc"],
        choices=["raw", "delta", "lpc"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="grid points to compress in parallel (the points are independent)",
    )
    p.add_argument("-o", "--out", type=Path, default=RESULTS_DIR / "compressors.json")
    args = p.parse_args()

    points = []
    for name in args.processes:
        for delta in sorted(args.deltas):
            if name in HARD and delta < args.hard_min_delta:
                print(f"--- {name}: SKIPPED delta={delta:g} (below --hard-min-delta)", flush=True)
            else:
                points.append((name, delta))

    jobs = min(args.jobs if args.jobs > 0 else os.cpu_count() or 1, len(points))
    runs = []
    if jobs > 1:
        names, deltas, configs = zip(*[(n, d, args) for n, d in points])
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            # map yields in the order submitted, so the reporting is ordered
            # even though the points finish out of order.
            for record in pool.map(run_point, names, deltas, configs):
                report(record)
                runs.append(record)
    else:
        for name, delta in points:
            record = run_point(name, delta, args)
            report(record)
            runs.append(record)

    config = {
        "processes": args.processes,
        "deltas": sorted(args.deltas),
        "hard_min_delta": args.hard_min_delta,
        "samples": args.samples,
        "repeats": args.repeats,
        "lpc_order": args.lpc_order,
        "methods": args.methods,
        "transforms": args.transforms,
        "seed": args.seed,
    }
    save_results(args.out, {"meta": run_meta("cpu"), "config": config, "runs": runs})


if __name__ == "__main__":
    main()
