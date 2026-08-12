#!/usr/bin/env python
"""Figures and tables for Section 5.3 from results/compressors*.json.

Writes

  * figures/fig_compressors.(png|pdf): bits per sample against the entropy
    rate as a function of the quantization step, for three families, with the
    five general-purpose coders shown as a band and the structure-aware
    coders as individual curves;
  * figures/fig_compressor_panel.(png|pdf): the overhead of every coder above
    the entropy rate, family by family, at delta = sigma/4 and delta = sigma;
  * results/compressors_table.md and ../paper/compressors_table.tex: the full
    panel at delta = sigma/4.

Several results files may be present (the fine steps of the strongly
smoothing families are run separately); they are merged by process and step,
later files overriding earlier ones codec by codec.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR

PAPER_TEX = Path(__file__).resolve().parent.parent / "paper" / "compressors_table.tex"

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LPC_COLOR = "#2a78d6"
FLAC_COLOR = "#1f8a70"
DELTA_ANS_COLOR = "#8a5fbf"
BAND_COLOR = "#eb6834"

#: (codec, transform, label) for the coders reported.  The general-purpose
#: byte compressors run on the raw stream, the common practice they stand for.
GENERAL = [
    ("zlib-9", "raw", "zlib -9"),
    ("zstd-19", "raw", "zstd -19"),
    ("brotli-11", "raw", "brotli -11"),
    ("lzma-9", "raw", "LZMA -9"),
    ("bz2-9", "raw", "bzip2 -9"),
]
STRUCTURED = [
    ("zlib-9", "delta", "delta coding + zlib -9"),
    ("ans", "delta", "delta coding + ANS"),
    ("flac-8", "raw", "FLAC -8"),
    ("ans", "lpc", "LPC(32) + ANS"),
]
#: The memoryless coder, which realizes the order-0 entropy and nothing more.
REFERENCE = [("ans", "raw", "ANS on raw")]

PANELS = [
    ("ma", "MA(8)"),
    ("ar1", "AR(1), $\\rho = 0.9$"),
    ("lowpass", "lowpass 6/30 kHz"),
]
NAME = {
    "white": "white",
    "diff": "first difference",
    "ma": "MA(8)",
    "ar1": "AR(1), $\\rho=0.9$",
    "gaussian1": "Gaussian, $\\tau=1$",
    "gaussian15": "Gaussian, $\\tau=1.5$",
    "lowpass": "lowpass 6/30 kHz",
}
SHORT = {
    "white": "white",
    "diff": "diff",
    "ma": "MA(8)",
    "ar1": "AR(1)",
    "gaussian1": "$\\tau{=}1$",
    "gaussian15": "$\\tau{=}1.5$",
    "lowpass": "lowpass",
}
ORDER = ["white", "diff", "ma", "ar1", "gaussian1", "gaussian15", "lowpass"]
DELTA_LABELS = {0.0625: "1/16", 0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}

#: Sequential ramp over the families, dark where the process carries the most
#: correlation structure (lowest entropy rate at a fixed step).  The markers
#: repeat the ordering, since seven shades are hard to tell apart in print.
RAMP = ["#cddff2", "#a3c8f0", "#7aaee8", "#5290da", "#2e6cbe", "#1a4d94", "#0d3a75"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

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
    """Every grid point, merged across results files."""
    merged: dict[tuple[str, float], dict] = {}
    for path in sorted(RESULTS_DIR.glob("compressors*.json")):
        for run in json.loads(path.read_text())["runs"]:
            key = (run["process"], run["delta"])
            if key in merged:
                merged[key]["codecs"].update(run["codecs"])
            else:
                merged[key] = run
    return list(merged.values())


def _bits(run: dict, method: str, transform: str) -> float:
    """Mean bits per sample of one codec on one transform of the stream.

    The lpc stage is stored under its resolved order, e.g. ``ans/lpc(32)``,
    and falls back to the raw stream when the predictor degenerated.
    """
    codecs = run["codecs"]
    key = next((k for k in codecs if k.startswith(f"{method}/{transform}")), None)
    if key is None and transform == "lpc":
        key = f"{method}/raw"
    if key is None:
        raise KeyError(
            f"{run['process']} delta={run['delta']:g}: no {method}/{transform} "
            f"among {sorted(codecs)}"
        )
    return float(np.mean(codecs[key]))


def _at_delta(runs: list[dict], delta: float) -> list[dict]:
    rows = [r for r in runs if r["delta"] == delta]
    rows.sort(key=lambda r: ORDER.index(r["process"]))
    return rows


def _overhead(run: dict, method: str, transform: str) -> float:
    """Percent above the entropy rate."""
    return 100.0 * (_bits(run, method, transform) - run["approx_bits"]) / run["approx_bits"]


def _coder_groups(runs: list[dict]) -> list[list[tuple[str, str, str]]]:
    """The three groups of coders, worst first within each, ranked by mean
    overhead at delta = sigma/4 so that figure and table read the same way."""
    rows = _at_delta(runs, 0.25)

    def mean_overhead(coder: tuple[str, str, str]) -> float:
        method, transform, _ = coder
        return float(np.mean([_overhead(r, method, transform) for r in rows]))

    return [
        REFERENCE,
        sorted(GENERAL, key=mean_overhead, reverse=True),
        sorted(STRUCTURED, key=mean_overhead, reverse=True),
    ]


def fig_compressors() -> None:
    """Bits per sample against the step, three families."""
    runs = _load()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7))

    for ax, (proc, title), panel in zip(axes, PANELS, "abc"):
        rows = sorted((r for r in runs if r["process"] == proc), key=lambda r: r["delta"])
        deltas = np.array([r["delta"] for r in rows])
        approx = np.array([r["approx_bits"] for r in rows])
        general = np.array(
            [[_bits(r, method, transform) for r in rows] for method, transform, _ in GENERAL]
        )
        ax.fill_between(
            deltas, general.min(axis=0), general.max(axis=0), color=BAND_COLOR,
            alpha=0.22, linewidth=0, label="general-purpose panel",
        )
        for (method, transform, label), color, marker in [
            (("flac-8", "raw", "FLAC -8"), FLAC_COLOR, "^"),
            (("ans", "delta", "delta coding + ANS"), DELTA_ANS_COLOR, "s"),
            (("ans", "lpc", "LPC(32) + ANS"), LPC_COLOR, "o"),
        ]:
            ax.plot(
                deltas, [_bits(r, method, transform) for r in rows], color=color,
                marker=marker, markersize=4.0, linewidth=1.6, label=label,
            )
        ax.plot(deltas, approx, color=INK, linewidth=1.4, linestyle="--", label="entropy rate")
        ax.set_xscale("log")
        ax.set_xticks(deltas)
        ax.set_xticklabels([DELTA_LABELS.get(d, f"{d:g}") for d in deltas])
        ax.minorticks_off()
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_xlabel("$\\Delta/\\sigma$")
        ax.set_title(f"({panel}) {title}", loc="left")
    axes[0].set_ylabel("bits per sample")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=8, handlelength=1.8,
        bbox_to_anchor=(0.5, -0.09),
    )
    fig.tight_layout()
    _write(fig, "fig_compressors")


def fig_compressor_panel() -> None:
    """Overhead above the entropy rate, coder by coder and family by family."""
    runs = _load()
    groups = _coder_groups(runs)
    style = {
        name: (RAMP[i], MARKERS[i]) for i, name in enumerate(_families_by_rate(runs))
    }

    # One row per coder, top to bottom, with a gap between the groups.
    placed: list[tuple[float, tuple[str, str, str]]] = []
    y = 0.0
    for group in groups:
        for coder in group:
            placed.append((y, coder))
            y -= 1.0
        y -= 0.6

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.8), sharey=True)
    for ax, delta, panel in zip(axes, (0.25, 1.0), "ab"):
        rows = _at_delta(runs, delta)
        for row_y, (method, transform, _) in placed:
            values = [_overhead(r, method, transform) for r in rows]
            ax.plot([min(values), max(values)], [row_y, row_y], color=GRID, linewidth=3.0,
                    solid_capstyle="round", zorder=1)
            for run, value in zip(rows, values):
                color, marker = style[run["process"]]
                ax.plot(value, row_y, marker=marker, markersize=5.0, markeredgewidth=0.6,
                        markeredgecolor="white", color=color, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100])
        ax.set_xticklabels(["1%", "3%", "10%", "30%", "100%"])
        ax.minorticks_off()
        ax.set_xlabel("overhead above the entropy rate")
        step = "\\sigma/4" if delta == 0.25 else "\\sigma"
        ax.set_title(f"({panel}) $\\Delta = {step}$", loc="left")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([row_y for row_y, _ in placed])
    axes[0].set_yticklabels([label for _, (_, _, label) in placed], fontsize=8)
    axes[0].set_ylim(placed[-1][0] - 0.8, 0.8)

    handles = [
        plt.Line2D([], [], marker=style[name][1], linestyle="none", markersize=5.0,
                   color=style[name][0], markeredgecolor="white", markeredgewidth=0.6,
                   label=NAME[name])
        for name in _families_by_rate(runs)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    _write(fig, "fig_compressor_panel")


def _families_by_rate(runs: list[dict]) -> list[str]:
    """Families ordered by entropy rate at delta = sigma/4, largest first, so
    that the color ramp darkens as the process becomes more compressible."""
    rows = _at_delta(runs, 0.25)
    return [r["process"] for r in sorted(rows, key=lambda r: -r["approx_bits"])]


def _write(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def compressors_table(delta: float = 0.25) -> None:
    """The full panel at one step, as markdown and as a LaTeX table body."""
    runs = _load()
    rows = _at_delta(runs, delta)
    groups = _coder_groups(runs)
    processes = [r["process"] for r in rows]

    def line(label: str, values: list[float], overhead: float | None) -> tuple[str, str]:
        cells = [f"{v:.3f}" for v in values]
        md_tail = "--" if overhead is None else f"{overhead:.0f}%"
        tex_tail = "--" if overhead is None else f"{overhead:.0f}\\%"
        md = f"| {label} | " + " | ".join(cells) + f" | {md_tail} |"
        tex = f"{label} & " + " & ".join(cells) + f" & {tex_tail} \\\\"
        return md, tex

    # The reference block, then one block per group of coders.
    blocks = [
        [
            line("entropy rate", [r["approx_bits"] for r in rows], None),
            line("order-0 entropy", [r["order0_bits"] for r in rows],
                 float(np.mean([100.0 * (r["order0_bits"] - r["approx_bits"]) / r["approx_bits"]
                                for r in rows]))),
        ]
    ]
    for group in groups:
        blocks.append([
            line(label, [_bits(r, method, transform) for r in rows],
                 float(np.mean([_overhead(r, method, transform) for r in rows])))
            for method, transform, label in group
        ])

    md = [
        "| coder | " + " | ".join(NAME[p] for p in processes) + " | mean overhead |",
        "| --- |" + " --: |" * (len(processes) + 1),
        *[m for block in blocks for m, _ in block],
    ]
    path = RESULTS_DIR / "compressors_table.md"
    path.write_text("\n".join(md) + "\n")
    print(f"wrote {path}")

    tex = [
        "{\\small",
        "\\begin{longtable}[]{@{}l" + "r" * (len(processes) + 1) + "@{}}",
        "\\toprule\\noalign{}",
        "coder & " + " & ".join(SHORT[p] for p in processes) + " & overhead \\\\",
        "\\midrule\\noalign{}",
        "\\endhead",
        "\\bottomrule\\noalign{}",
        "\\endlastfoot",
    ]
    for i, block in enumerate(blocks):
        if i:
            tex.append("\\midrule\\noalign{}")
        tex.extend(t for _, t in block)
    tex += ["\\end{longtable}", "}"]
    PAPER_TEX.write_text("\n".join(tex) + "\n")
    print(f"wrote {PAPER_TEX}")


def summary(delta: float = 0.25) -> None:
    """Ranges quoted in the text, printed rather than written to a file."""
    runs = _load()
    coders = [coder for group in _coder_groups(runs) for coder in group]
    for step in (0.25, 1.0):
        rows = _at_delta(runs, step)
        print(f"\ndelta = {step:g} sigma")
        for method, transform, label in coders:
            over = [_overhead(r, method, transform) for r in rows]
            worst = rows[int(np.argmax(over))]["process"]
            best = rows[int(np.argmin(over))]["process"]
            # No lossless coder can go below the entropy rate; a negative
            # overhead would mean the reference rate is off, not that the
            # coder won.
            flag = "   BELOW THE RATE" if min(over) < 0 else ""
            print(f"  {label:>22}  {min(over):6.1f}% ({best})  to {max(over):6.1f}% ({worst})"
                  f"   mean {np.mean(over):6.1f}%{flag}")

    # Where FLAC's remaining gap sits: give its coder an already whitened
    # stream and compare against the same residual coded by ANS.
    print(f"\nFLAC on the LPC(32) residual, delta = {delta:g} sigma (bits/sample)")
    for run in _at_delta(runs, delta):
        print(f"  {NAME[run['process']]:>20}  rate {run['approx_bits']:.3f}"
              f"   FLAC {_bits(run, 'flac-8', 'raw'):.3f}"
              f"   FLAC on the residual {_bits(run, 'flac-8', 'lpc'):.3f}"
              f"   ANS on the residual {_bits(run, 'ans', 'lpc'):.3f}")


if __name__ == "__main__":
    fig_compressors()
    fig_compressor_panel()
    compressors_table()
    summary()
