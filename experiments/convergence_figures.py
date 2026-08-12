#!/usr/bin/env python
"""Figures and tables for Section 5.1 from the JSON results in results/.

Reads convergence_particles*.json, convergence_length*.json and convergence_collapse*.json
(several files per experiment are merged, so that a family added later can be
run on its own) and writes to figures/:

  fig_convergence_particles.(png|pdf)   estimate vs N, and paired excess vs N
  fig_convergence_length.(png|pdf)      estimate vs n, and replicate scatter vs n
  fig_convergence_collapse.(png|pdf)    lost-lock fraction and ESS vs N

and to results/: convergence_collapse_table.md, the tables for the collapse study.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from common import FIGURES_DIR, RESULTS_DIR

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
#: Ordered families as lightness ramps (MA blues by width, Gaussian oranges
#: by tau), the first difference in purple; distinct markers throughout.
SERIES_COLOR = {
    "ma": "#8fb9f0",
    "ma16": "#4f95e6",
    "ma32": "#2261b8",
    "ma64": "#0d3a75",
    "gaussian1": "#c26d24",
    "gaussian125": "#8a4a14",
    "diff": "#7b4fa6",
    "ar1": "#eb6834",
}
SERIES_MARKER = {
    "ma": "o", "ma16": "s", "ma32": "P", "ma64": "X",
    "gaussian1": "^", "gaussian125": "v", "diff": "D", "ar1": "s",
}
SERIES_NAME = {
    "ma": "MA(8)",
    "ma16": "MA(16)",
    "ma32": "MA(32)",
    "ma64": "MA(64)",
    "ar1": "AR(1), $\\rho=0.9$",
    "diff": "first difference",
    "gaussian1": "Gaussian, $\\tau=1$",
    "gaussian125": "Gaussian, $\\tau=1.25$",
    "gaussian15": "Gaussian, $\\tau=1.5$",
    "lowpass": "lowpass 6 kHz / 30 kHz",
}
#: Plot order top-to-bottom by bias size, so the legend reads like the plot.
SERIES_ORDER = ["ma64", "ma32", "ma16", "ma", "gaussian125", "gaussian1", "diff"]

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

ERRORBAR_STYLE = dict(markersize=4.5, linewidth=1.8, capsize=2.5, capthick=1.0, elinewidth=1.0)


def _load_runs(stem: str) -> list[dict]:
    """All runs of one experiment, merged over convergence_<stem>*.json."""
    paths = sorted(RESULTS_DIR.glob(f"convergence_{stem}*.json"))
    if not paths:
        raise FileNotFoundError(f"no results matching convergence_{stem}*.json in {RESULTS_DIR}")
    runs: list[dict] = []
    for path in paths:
        runs.extend(json.loads(path.read_text())["runs"])
    return runs


def _kfmt(value, _pos=None) -> str:
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def _save(fig, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def _by_process(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["process"], []).append(run)
    ordered = [k for k in SERIES_ORDER if k in grouped]
    ordered += [k for k in grouped if k not in ordered]
    return {k: grouped[k] for k in ordered}


def _repeat_values(run: dict) -> np.ndarray:
    """Per-replicate estimates ordered by replicate index (usable ones only)."""
    reps = sorted(run["repeats"], key=lambda u: u["index"])
    return np.array(
        [u["entropy_rate_bits"] if u["usable"] else np.nan for u in reps], dtype=float
    )


def _log_x(ax, ticks: np.ndarray) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax.set_xticks(np.asarray(ticks, dtype=float))
    ax.minorticks_off()


def fig_particles() -> None:
    """Paired excess over the largest N, log-log, with the 1/N guide.

    The largest particle count of each process serves as the reference; each
    curve is drawn up to the last N at which the paired excess is resolved
    (mean above two standard errors), beyond which the excess is lost in the
    filter noise.
    """
    grouped = {k: v for k, v in _by_process(_load_runs("particles")).items() if k in SERIES_ORDER}
    fig, ax = plt.subplots(figsize=(4.6, 3.4))

    ticks: set[float] = set()
    for name, runs in grouped.items():
        runs = sorted(runs, key=lambda r: r["result"]["n_particles"])
        grid = np.array([r["result"]["n_particles"] for r in runs])

        # Paired excess over the largest-N run: the same replicate index sees
        # the same observed sequence at every N, so the difference removes the
        # sequence-to-sequence variation and isolates the filter bias.
        ref = _repeat_values(runs[-1])
        excess_mean, excess_se = [], []
        for run in runs[:-1]:
            diff = _repeat_values(run) - ref
            diff = diff[np.isfinite(diff)]
            excess_mean.append(diff.mean())
            excess_se.append(diff.std(ddof=1) / np.sqrt(diff.size))
        excess_mean = np.array(excess_mean)
        excess_se = np.array(excess_se)

        resolved = excess_mean > 2.0 * excess_se
        keep = 0
        while keep < resolved.size and resolved[keep]:
            keep += 1
        if keep == 0:
            print(f"note: {name}: excess unresolved even at the smallest N; skipped")
            continue
        ticks.update(grid[:keep].tolist())
        ax.errorbar(
            grid[:keep], excess_mean[:keep], yerr=excess_se[:keep],
            color=SERIES_COLOR[name], marker=SERIES_MARKER[name],
            label=SERIES_NAME[name], **ERRORBAR_STYLE,
        )

    # A short 1/N guide in the clear region below the lowest curve.
    ticks = np.array(sorted(ticks))
    anchor = min(line.get_ydata()[0] for line in ax.get_lines())
    guide_x = np.array([ticks[0], ticks[3]])
    guide_y = 0.5 * anchor * guide_x[0] / guide_x
    ax.plot(guide_x, guide_y, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(
        "$\\propto 1/N$", (guide_x[1], guide_y[1]),
        textcoords="offset points", xytext=(4, -2), color=INK2, fontsize=8,
    )

    _log_x(ax, ticks)
    ax.set_yscale("log")
    _style(ax, "particles $N$", "excess over largest $N$ (bits)")
    legend = ax.legend(
        loc="upper right", handlelength=1.6, fontsize=7.5, labelspacing=0.35,
        frameon=True, framealpha=0.9, edgecolor="none", facecolor="white",
    )
    legend.set_zorder(10)
    fig.tight_layout()
    _save(fig, "fig_convergence_particles")


def fig_length() -> None:
    """Replicate standard deviation against the sequence length, log-log."""
    grouped = {k: v for k, v in _by_process(_load_runs("length")).items() if k in SERIES_ORDER}
    fig, ax = plt.subplots(figsize=(4.6, 3.4))

    ticks: set[float] = set()
    for name, runs in grouped.items():
        runs = sorted(runs, key=lambda r: r["result"]["n"])
        grid = np.array([r["result"]["n"] for r in runs])
        ticks.update(grid.tolist())
        sd = np.array([np.nanstd(_repeat_values(r), ddof=1) for r in runs])
        counts = np.array([np.isfinite(_repeat_values(r)).sum() for r in runs])
        # Standard error of a standard deviation from r replicates.
        sd_se = sd / np.sqrt(2.0 * (counts - 1))
        ax.errorbar(
            grid, sd, yerr=sd_se,
            color=SERIES_COLOR[name], marker=SERIES_MARKER[name],
            label=SERIES_NAME[name], **ERRORBAR_STYLE,
        )

    ticks = np.array(sorted(ticks))
    anchor = max(line.get_ydata()[0] for line in ax.get_lines())
    guide_y = 2.2 * anchor * np.sqrt(ticks[0] / ticks)
    ax.plot(ticks, guide_y, color=MUTED, linewidth=1.0, linestyle="--")
    ax.annotate(
        "$\\propto 1/\\sqrt{n}$", (ticks[-1], guide_y[-1]),
        textcoords="offset points", xytext=(4, 0), color=INK2, fontsize=8,
    )

    _log_x(ax, ticks)
    ax.set_yscale("log")
    _style(ax, "sequence length $n$", "replicate std. dev. (bits)")
    legend = ax.legend(
        loc="upper right", handlelength=1.6, fontsize=7.5, labelspacing=0.35,
        frameon=True, framealpha=0.9, edgecolor="none", facecolor="white",
    )
    legend.set_zorder(10)
    fig.tight_layout()
    _save(fig, "fig_convergence_length")


def fig_collapse() -> None:
    grouped = _by_process(_load_runs("collapse"))
    fig, axes = plt.subplots(1, len(grouped), figsize=(3.3 * len(grouped), 2.7), squeeze=False)

    for ax, (name, runs), panel in zip(axes[0], grouped.items(), "abcd"):
        runs = sorted(runs, key=lambda r: r["result"]["n_particles"])
        grid = np.array([r["result"]["n_particles"] for r in runs])
        lost = 100.0 * np.array([r["result"]["lost_lock_fraction"] for r in runs])
        ess = 100.0 * np.array([r["result"]["mean_ess"] for r in runs])

        ax.plot(
            grid, ess, color=MUTED, marker="s", markersize=4.5, linewidth=1.8,
            label="mean ESS (% of $N$)",
        )
        ax.plot(
            grid, lost, color="#2a78d6", marker="o", markersize=4.5, linewidth=1.8,
            label="lost-lock steps (% of steps)",
        )
        ax.legend(loc="lower left", handlelength=1.6, fontsize=8)
        _log_x(ax, grid)
        ax.set_ylim(-3, 103)
        delta_h0 = runs[0]["result"]["delta"] / runs[0]["result"]["h0"]
        _style(ax, "particles $N$", "% of steps / % of $N$")
        ax.set_title(
            f"({panel}) {SERIES_NAME.get(name, name)}, $\\Delta/h_0 = {delta_h0:.1f}$",
            loc="left",
        )
    fig.tight_layout()
    _save(fig, "fig_convergence_collapse")


def collapse_table() -> None:
    grouped = _by_process(_load_runs("collapse"))
    lines: list[str] = []
    for name, runs in grouped.items():
        runs = sorted(runs, key=lambda r: r["result"]["n_particles"])
        res0 = runs[0]["result"]
        lines += [
            f"**{SERIES_NAME.get(name, name)}** "
            f"($\\Delta = {res0['delta']:g}\\sigma$, $\\Delta/h_0 = {res0['delta'] / res0['h0']:.1f}$, "
            f"$L = {res0['n_taps']}$, $n = {res0['n']}$):",
            "",
            "| $N$ | estimate (bits/sample) | discarded replicates | lost-lock steps | mean ESS |",
            "| --: | --: | --: | --: | --: |",
        ]
        for run in runs:
            res = run["result"]
            n_failed = res["n_failed_repeats"]
            if res["n_usable_repeats"] == 0:
                estimate = f"{res['entropy_rate_bits']:.3g} (collapsed)"
            elif n_failed:
                estimate = f"{res['entropy_rate_bits']:.4f} (provisional)"
            else:
                estimate = f"{res['entropy_rate_bits']:.4f} $\\pm$ {res['stderr_bits']:.4f}"
            lines.append(
                f"| {res['n_particles']} | {estimate} | {n_failed}/{res['n_repeats']} "
                f"| {100 * res['lost_lock_fraction']:.1f}% | {100 * res['mean_ess']:.0f}% |"
            )
        lines.append("")
    path = RESULTS_DIR / "convergence_collapse_table.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    fig_particles()
    fig_length()
    fig_collapse()
    collapse_table()
