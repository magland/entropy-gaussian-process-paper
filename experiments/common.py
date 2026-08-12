"""Shared infrastructure for the Section 5 experiment scripts.

Each experiment script builds its process specifications with the helpers
here, runs :func:`egp.estimate_entropy_rate` over a parameter grid, and
records one JSON file holding every estimate together with the per-replicate
stream and enough metadata to rerun the sweep.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import egp

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"

#: The processes used in the experiments, all normalized to sigma = 1.
PROCESS_BUILDERS = {
    "white": lambda: egp.from_preset("white"),
    "diff": lambda: egp.from_preset("diff"),
    "ma": lambda: egp.from_preset("ma", taps=8),
    "ma16": lambda: egp.from_preset("ma", taps=16),
    "ma32": lambda: egp.from_preset("ma", taps=32),
    "ma64": lambda: egp.from_preset("ma", taps=64),
    "ar1": lambda: egp.from_preset("ar1", rho=0.9),
    "gaussian1": lambda: egp.from_preset("gaussian", tau=1.0, taps=9),
    "gaussian125": lambda: egp.from_preset("gaussian", tau=1.25, taps=11),
    "gaussian15": lambda: egp.from_preset("gaussian", tau=1.5, taps=13),
    "lowpass": lambda: egp.from_preset("lowpass", cutoff=6000.0, fs=30000.0, taps=33),
}


def process_spec(name: str) -> egp.ProcessSpec:
    try:
        builder = PROCESS_BUILDERS[name]
    except KeyError:
        raise ValueError(f"unknown process {name!r}; choose from {sorted(PROCESS_BUILDERS)}") from None
    return builder()


def run_meta(engine: str) -> dict:
    """Provenance recorded at the top of every results file."""
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "egp_version": egp.__version__,
        "engine": engine,
        "argv": sys.argv,
    }
    if engine == "gpu":
        from egp.pf_gpu import gpu_description

        meta["gpu"] = gpu_description()
    return meta


def run_case(
    spec: egp.ProcessSpec,
    delta: float,
    *,
    n: int,
    n_particles: int,
    n_repeats: int,
    seed: int,
    engine: str,
    tag: str = "",
) -> dict:
    """One grid point: run the estimator and return a JSON-ready record."""
    label = tag or spec.label
    print(f"--- {label}  delta={delta:g}  n={n}  N={n_particles}  r={n_repeats}", flush=True)
    repeats: list[dict] = []

    def on_repeat(u: egp.RepeatUpdate) -> None:
        repeats.append(
            {
                "index": u.index,
                "entropy_rate_bits": u.entropy_rate_bits,
                "mean_ess": u.mean_ess,
                "lost_lock_fraction": u.lost_lock_fraction,
                "usable": u.usable,
                "seconds": u.seconds,
            }
        )
        flag = "" if u.usable else "   LOST LOCK"
        print(
            f"    [{u.index + 1}/{u.n_repeats}] {u.entropy_rate_bits:14.4f} bits"
            f"   mean {u.running_mean_bits:.4f} +/- {u.running_stderr_bits:.4f}"
            f"   ({u.seconds:.1f}s){flag}",
            flush=True,
        )

    started = time.time()
    result = egp.estimate_entropy_rate(
        spec,
        delta,
        n=n,
        n_particles=n_particles,
        n_repeats=n_repeats,
        seed=seed,
        engine=engine,
        on_repeat=on_repeat,
    )
    print(
        f"    -> {result.entropy_rate_bits:.4f} +/- {result.stderr_bits:.4f} bits"
        f"   approx {result.approximation_bits:.4f}"
        f"   failed {result.n_failed_repeats}/{result.n_repeats}"
        f"   ({time.time() - started:.0f}s)",
        flush=True,
    )
    return {"result": result.to_dict(), "repeats": repeats}


def save_results(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def load_family_curves() -> dict:
    """Merge every family_curves*.json; later files override point by point.

    Returns panels keyed by family name; each panel carries its dense
    approximation grid and the list of Monte Carlo runs.
    """
    panels: dict = {}
    for path in sorted(RESULTS_DIR.glob("family_curves*.json")):
        for name, panel in json.loads(path.read_text())["panels"].items():
            if name not in panels:
                panels[name] = panel
                continue
            merged = panels[name]
            if "dense" in panel:
                merged["dense"] = panel["dense"]
            points = {(run["value"], run["result"]["delta"]): run for run in merged["mc"]}
            for run in panel["mc"]:
                points[(run["value"], run["result"]["delta"])] = run
            merged["mc"] = list(points.values())
    return panels
