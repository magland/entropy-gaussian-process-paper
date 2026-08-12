#!/usr/bin/env python
"""Figure and table for Section 5.3 from results/compressors.json.

Writes figures/fig_compressors.(png|pdf), bits per sample of the two
representative coders against the entropy rate for two families, and
results/compressors_table.md, all families at delta = sigma/4.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LPC_COLOR = "#2a78d6"
ZLIB_COLOR = "#eb6834"

PANELS = [("ar1", "AR(1), $\\rho = 0.9$"), ("lowpass", "lowpass 6/30 kHz")]
NAME = {
    "white": "white",
    "diff": "first difference",
    "ma": "MA(8)",
    "ar1": "AR(1), $\\rho=0.9$",
    "gaussian1": "Gaussian, $\\tau=1$",
    "gaussian15": "Gaussian, $\\tau=1.5$",
    "lowpass": "lowpass 6/30 kHz",
}
ORDER = ["white", "diff", "ma", "ar1", "gaussian1", "gaussian15", "lowpass"]
DELTA_LABELS = {0.0625: "1/16", 0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _load() -> list[dict]:
    runs: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("compressors*.json")):
        runs.extend(json.loads(path.read_text())["runs"])
    return runs


def _stage(run: dict, method: str, transform: str) -> float:
    """Mean bits/sample of one codec/stage; the lpc stage falls back to raw
    when the predictor degenerated (white noise at order 0)."""
    codecs = run["codecs"]
    key = next((k for k in codecs if k.startswith(f"{method}/{transform}")), None)
    if key is None and transform == "lpc":
        key = f"{method}/raw"
    return float(np.mean(codecs[key]))


def fig_compressors() -> None:
    runs = _load()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    for ax, (proc, title), panel in zip(axes, PANELS, "ab"):
        rows = sorted((r for r in runs if r["process"] == proc), key=lambda r: r["delta"])
        deltas = np.array([r["delta"] for r in rows])
        approx = np.array([r["approx_bits"] for r in rows])
        lpc = np.array([_stage(r, "ans", "lpc") for r in rows])
        zlib = np.array([_stage(r, "zlib-9", "raw") for r in rows])
        ax.plot(deltas, zlib, color=ZLIB_COLOR, marker="s", markersize=4.5,
                linewidth=1.8, label="zlib on raw stream")
        ax.plot(deltas, lpc, color=LPC_COLOR, marker="o", markersize=4.5,
                linewidth=1.8, label="LPC(32) + ANS")
        ax.plot(deltas, approx, color=INK, linewidth=1.4, linestyle="--",
                label="entropy rate")
        ax.set_xscale("log")
        ax.set_xticks(deltas)
        ax.set_xticklabels([DELTA_LABELS.get(d, f"{d:g}") for d in deltas])
        ax.minorticks_off()
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xlabel("$\\Delta/\\sigma$")
        ax.set_title(f"({panel}) {title}", loc="left")
    axes[0].set_ylabel("bits per sample")
    axes[0].legend(loc="upper right", handlelength=1.8, fontsize=8)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"fig_compressors.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def compressors_table(delta: float = 0.25) -> None:
    runs = [r for r in _load() if r["delta"] == delta]
    runs.sort(key=lambda r: ORDER.index(r["process"]))
    lines = [
        "| process | entropy rate | order-0 | LPC(32) + ANS | zlib on raw |",
        "| --- | --: | --: | --: | --: |",
    ]
    for run in runs:
        approx = run["approx_bits"]
        lpc = _stage(run, "ans", "lpc")
        zlib = _stage(run, "zlib-9", "raw")
        lines.append(
            f"| {NAME[run['process']]} | {approx:.3f} | {run['order0_bits']:.3f} "
            f"| {lpc:.3f} (+{100 * (lpc - approx) / approx:.1f}%) "
            f"| {zlib:.3f} (+{100 * (zlib - approx) / approx:.0f}%) |"
        )
    path = RESULTS_DIR / "compressors_table.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    fig_compressors()
    compressors_table()
