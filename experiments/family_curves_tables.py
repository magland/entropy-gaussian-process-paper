#!/usr/bin/env python
"""Appendix tables for Section 5.2 from results/family_curves*.json.

Writes the detailed comparison tables (particle-filter estimate against the
analytical approximation at every Monte Carlo point of the family sweeps) as

  * ../paper/family_curves_tables.tex, input by paper.tex in the appendix;
  * results/family_curves_tables.md, for quick inspection.

Table numbering (C1, C2, ...) is emitted here, matching the manual table
numbering used in the paper body.
"""

from __future__ import annotations

from pathlib import Path

from common import RESULTS_DIR, load_family_curves

PAPER_TEX = Path(__file__).resolve().parent.parent / "paper" / "family_curves_tables.tex"

FAMILY_INFO = {
    "ma": ("moving average", "width $w$"),
    "ar1": ("AR(1), 64 taps", "correlation $\\rho$"),
    "gaussian": ("Gaussian kernel", "width $\\tau$"),
    "lowpass": ("lowpass, 33 taps", "cutoff / $f_s$"),
}
PROCESS_LABEL = {"white": "white noise", "diff": "first difference"}


def fmt_value(v: float) -> str:
    return f"{v:g}"


def fmt_delta(d: float) -> str:
    return f"{d:g}"


def rows_for(panel: dict) -> list[dict]:
    rows = []
    for run in sorted(panel["mc"], key=lambda r: (r["value"], r["result"]["delta"])):
        res = run["result"]
        collapsed = res["n_usable_repeats"] == 0
        rows.append(
            {
                "value": run["value"],
                "delta": res["delta"],
                "N": res["n_particles"],
                "n": res["n"],
                "r": res["n_repeats"],
                "estimate": None if collapsed else res["entropy_rate_bits"],
                "stderr": None if collapsed else res["stderr_bits"],
                "approx": res["approximation_bits"],
                "discarded": res["n_failed_repeats"],
                "repeats": res["n_repeats"],
            }
        )
    return rows


def estimate_cell(row: dict, latex: bool) -> str:
    if row["estimate"] is None:
        return "collapsed"
    pm = "$\\pm$" if latex else "+/-"
    cell = f"{row['estimate']:.4f} {pm} {row['stderr']:.4f}"
    if row["discarded"]:
        cell += f" ({row['discarded']}/{row['repeats']} discarded)"
    return cell


def error_cell(row: dict) -> str:
    if row["estimate"] is None:
        return "--"
    return f"{1000.0 * (row['approx'] - row['estimate']):+.1f}"


def latex_table(number: int, heading: str, col0: str, rows: list[dict], note: str) -> str:
    lines = [
        f"\\textbf{{Table C{number}.}} {heading} {note}",
        "",
        "{\\small",
        "\\begin{longtable}[]{@{}rrrrrrrr@{}}",
        "\\toprule\\noalign{}",
        f"{col0} & $\\Delta/\\sigma$ & $N$ & $n$ & $r$ & estimate (bits/sample) & approximation & error (millibits) \\\\",
        "\\midrule\\noalign{}",
        "\\endhead",
        "\\bottomrule\\noalign{}",
        "\\endlastfoot",
    ]
    for row in rows:
        lines.append(
            f"{fmt_value(row['value'])} & {fmt_delta(row['delta'])} & {row['N']} & "
            f"{row['n']} & {row['r']} & "
            f"{estimate_cell(row, latex=True)} & {row['approx']:.4f} & {error_cell(row)} \\\\"
        )
    lines += ["\\end{longtable}", "}", ""]
    return "\n".join(lines)


def markdown_table(heading: str, col0: str, rows: list[dict]) -> str:
    lines = [
        f"## {heading}",
        "",
        f"| {col0} | delta/sigma | N | n | r | estimate (bits/sample) | approximation | error (mb) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {fmt_value(row['value'])} | {fmt_delta(row['delta'])} | {row['N']} | "
            f"{row['n']} | {row['r']} | "
            f"{estimate_cell(row, latex=False)} | {row['approx']:.4f} | {error_cell(row)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    panels = load_family_curves()
    tex_parts = []
    md_parts = ["# Section 5.2 comparison tables\n"]
    number = 1

    for name, (title, param_label) in FAMILY_INFO.items():
        if name not in panels:
            continue
        rows = rows_for(panels[name])
        note = (
            "The particle-filter estimate against the analytical approximation "
            "(error is approximation minus estimate), with the particle count $N$, "
            "sequence length $n$, and replicate count $r$ used at each point."
        )
        tex_parts.append(latex_table(number, f"\\textbf{{{title}}}.", param_label, rows, note))
        md_parts.append(markdown_table(f"{title}", param_label.replace("$", ""), rows))
        number += 1

    delta_rows = []
    for name in ("white", "diff"):
        if name not in panels:
            continue
        for row in rows_for(panels[name]):
            row["value_label"] = PROCESS_LABEL[name]
            delta_rows.append(row)
    if delta_rows:
        lines = [
            f"\\textbf{{Table C{number}.}} \\textbf{{White noise and first difference}}. "
            "The particle-filter estimate against the analytical approximation over "
            "the quantization step (error is approximation minus estimate).",
            "",
            "{\\small",
            "\\begin{longtable}[]{@{}lrrrrrrr@{}}",
            "\\toprule\\noalign{}",
            "process & $\\Delta/\\sigma$ & $N$ & $n$ & $r$ & estimate (bits/sample) & approximation & error (millibits) \\\\",
            "\\midrule\\noalign{}",
            "\\endhead",
            "\\bottomrule\\noalign{}",
            "\\endlastfoot",
        ]
        md = [
            "## white noise and first difference",
            "",
            "| process | delta/sigma | N | n | r | estimate (bits/sample) | approximation | error (mb) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in delta_rows:
            common = (
                f"{fmt_delta(row['delta'])} & {row['N']} & {row['n']} & {row['r']} & "
                f"{estimate_cell(row, latex=True)} & {row['approx']:.4f} & {error_cell(row)}"
            )
            lines.append(f"{row['value_label']} & {common} \\\\")
            md.append(
                f"| {row['value_label']} | {fmt_delta(row['delta'])} | {row['N']} | "
                f"{row['n']} | {row['r']} | "
                f"{estimate_cell(row, latex=False)} | {row['approx']:.4f} | {error_cell(row)} |"
            )
        lines += ["\\end{longtable}", "}", ""]
        tex_parts.append("\n".join(lines))
        md_parts.append("\n".join(md))

    PAPER_TEX.write_text("\n".join(tex_parts))
    print(f"wrote {PAPER_TEX}")
    md_path = RESULTS_DIR / "family_curves_tables.md"
    md_path.write_text("\n".join(md_parts))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
