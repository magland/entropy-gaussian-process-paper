#!/usr/bin/env python
"""Section 5.2 figures from results/family_curves*.json.

Writes one figure per family, figures/fig_family_<name>.(png|pdf), each with
two panels: (a) the entropy rate, the analytical approximation as dense
curves with the particle-filter estimates overlaid as open circles; (b) the
error of the approximation (approximation minus estimate, in millibits per
sample) at the estimated points, with one-standard-error bars.

For the four parameterized families the x axis is the family parameter and
there is one curve per quantization step.  Each error panel is scaled to its
own family.  For white noise and the first difference,
which have no family parameter, the x axis is the quantization step itself
and the two processes appear together in one figure.

Several results files may be present (e.g. per-family reruns at larger N);
they are merged, later files overriding earlier ones point by point.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, load_family_curves

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LOST = "#b0342b"

#: Sequential blue ramp, dark at the finest step (where the rate is largest).
DELTA_COLOR = {0.125: "#0d3a75", 0.25: "#2e6cbe", 1.0: "#5f9ce4", 2.0: "#a3c8f0"}
DELTA_LABEL = {
    0.125: "$\\Delta = \\sigma/8$",
    0.25: "$\\Delta = \\sigma/4$",
    1.0: "$\\Delta = \\sigma$",
    2.0: "$\\Delta = 2\\sigma$",
}
DELTA_MARKER = {0.125: "o", 0.25: "s", 1.0: "D", 2.0: "^"}

#: Categorical pair for the delta-sweep figure (validated for CVD).
PROCESS_COLOR = {"white": "#2e6cbe", "diff": "#c26d24"}
PROCESS_LABEL = {"white": "white noise", "diff": "first difference"}
PROCESS_MARKER = {"white": "o", "diff": "s"}

FAMILY_STYLE = {
    "ma": ("moving average", "width $w$", True),
    "ar1": ("AR(1), 64 taps", "correlation $\\rho$", False),
    "gaussian": ("Gaussian kernel", "width $\\tau$", False),
    "lowpass": ("lowpass (33 taps)", "cutoff / $f_s$", False),
}
PARAM_ORDER = ["ma", "ar1", "gaussian", "lowpass"]

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


def _finish(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES_DIR / f"fig_family_{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def _legend(fig, handles) -> None:
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.955),
        ncol=len(handles), fontsize=7, columnspacing=0.9, handlelength=1.5,
        handletextpad=0.5,
    )


def plot_param_family(name: str, panel: dict) -> None:
    title, xlabel, log_x = FAMILY_STYLE[name]
    deltas = panel["deltas"]
    dense_values = np.array(panel["dense"]["values"])
    dense_rates = np.array(panel["dense"]["rates"])  # [len(values)][len(deltas)]

    fig, (ax_rate, ax_err) = plt.subplots(1, 2, figsize=(6.9, 2.9))

    for j, delta in enumerate(deltas):
        ax_rate.plot(
            dense_values, dense_rates[:, j],
            color=DELTA_COLOR[delta], linewidth=1.8,
        )

    lost = []
    for run in panel["mc"]:
        res = run["result"]
        delta = res["delta"]
        if res["n_usable_repeats"] == 0:
            lost.append((run["value"], res["approximation_bits"]))
            continue
        ax_rate.errorbar(
            [run["value"]], [res["entropy_rate_bits"]],
            yerr=[res["stderr_bits"]],
            linestyle="none", marker="o", markersize=4.5,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0,
            ecolor=INK, elinewidth=0.8, capsize=0, zorder=5,
        )
        err_mb = 1000.0 * (res["approximation_bits"] - res["entropy_rate_bits"])
        se_mb = 1000.0 * res["stderr_bits"]
        ax_err.errorbar(
            [run["value"]], [err_mb], yerr=[se_mb],
            linestyle="none", marker=DELTA_MARKER[delta], markersize=4,
            color=DELTA_COLOR[delta], elinewidth=0.9, capsize=0, zorder=5,
        )
    if lost:
        xs, ys = zip(*lost)
        ax_rate.plot(
            xs, ys, linestyle="none", marker="x", markersize=4.5,
            color=LOST, markeredgewidth=1.1, zorder=6,
        )
        print(f"{name}: {len(lost)} point(s) with no usable replicate")

    ax_err.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)

    for ax in (ax_rate, ax_err):
        if log_x:
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
            ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    ax_rate.set_ylabel("entropy rate (bits/sample)")
    ax_rate.set_ylim(0, 6.4)
    ax_rate.set_title("(a)", loc="left")
    ax_err.set_ylabel("approx. $-$ estimate (millibits)")
    ax_err.set_title("(b)", loc="left")

    handles = [
        plt.Line2D(
            [], [], color=DELTA_COLOR[d], linewidth=1.8, marker=DELTA_MARKER[d],
            markersize=4, label=DELTA_LABEL[d],
        )
        for d in deltas
    ]
    handles.append(
        plt.Line2D(
            [], [], linestyle="none", marker="o", markersize=4.5,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0,
            label="particle-filter estimate",
        )
    )
    if lost:
        handles.append(
            plt.Line2D(
                [], [], linestyle="none", marker="x", markersize=4.5,
                color=LOST, markeredgewidth=1.1, label="estimator lost lock",
            )
        )
    _legend(fig, handles)
    fig.suptitle(title, x=0.02, y=1.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _finish(fig, name)


def plot_delta_sweep(panels: dict) -> None:
    fig, (ax_rate, ax_err) = plt.subplots(1, 2, figsize=(6.9, 2.9))

    for name in ("white", "diff"):
        panel = panels[name]
        color = PROCESS_COLOR[name]
        ax_rate.plot(
            panel["dense"]["values"], panel["dense"]["rates"],
            color=color, linewidth=1.8,
        )
        for run in panel["mc"]:
            res = run["result"]
            if res["n_usable_repeats"] == 0:
                continue
            ax_rate.errorbar(
                [run["value"]], [res["entropy_rate_bits"]],
                yerr=[res["stderr_bits"]],
                linestyle="none", marker="o", markersize=4.5,
                markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0,
                ecolor=INK, elinewidth=0.8, capsize=0, zorder=5,
            )
            err_mb = 1000.0 * (res["approximation_bits"] - res["entropy_rate_bits"])
            se_mb = 1000.0 * res["stderr_bits"]
            ax_err.errorbar(
                [run["value"]], [err_mb], yerr=[se_mb],
                linestyle="none", marker=PROCESS_MARKER[name], markersize=4,
                color=color, elinewidth=0.9, capsize=0, zorder=5,
            )

    ax_err.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ticks = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    labels = ["1/8", "1/4", "1/2", "1", "2", "4"]
    for ax in (ax_rate, ax_err):
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.minorticks_off()
        ax.set_xlabel("$\\Delta/\\sigma$")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    ax_rate.set_ylabel("entropy rate (bits/sample)")
    ax_rate.set_ylim(0, 6.4)
    ax_rate.set_title("(a)", loc="left")
    ax_err.set_ylabel("approx. $-$ estimate (millibits)")
    ax_err.set_title("(b)", loc="left")

    handles = [
        plt.Line2D(
            [], [], color=PROCESS_COLOR[name], linewidth=1.8,
            marker=PROCESS_MARKER[name], markersize=4, label=PROCESS_LABEL[name],
        )
        for name in ("white", "diff")
    ]
    handles.append(
        plt.Line2D(
            [], [], linestyle="none", marker="o", markersize=4.5,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0,
            label="particle-filter estimate",
        )
    )
    _legend(fig, handles)
    fig.suptitle("white noise and first difference", x=0.02, y=1.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _finish(fig, "whitediff")


def main() -> None:
    panels = load_family_curves()
    if not panels:
        raise SystemExit("no results/family_curves*.json found; run family_curves.py first")
    for name in PARAM_ORDER:
        if name in panels:
            plot_param_family(name, panels[name])
    if "white" in panels and "diff" in panels:
        plot_delta_sweep(panels)


if __name__ == "__main__":
    main()
