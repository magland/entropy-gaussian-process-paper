# Entropy rate of a quantized stationary Gaussian process

How many bits per sample does a rounded Gaussian signal really carry?

Let $X_t$ be a zero-mean stationary Gaussian process and $Y_t =
\operatorname{round}(X_t/\Delta)$ its uniform quantization. The entropy rate
$\bar H$ of $\{Y_t\}$ is the limit no lossless code can beat, but it has no
closed form: the likelihood of a block is an $n$-dimensional Gaussian integral
over a box. This repository contains a paper developing two methods for
computing it, a reference implementation, an interactive demonstration, and
the scripts that reproduce every figure in the paper.

The two methods, in one line each. The analytical approximation is the
classical high-resolution formula with the quantizer's own noise power
$\Delta^2/12$ added to the spectrum, which keeps it finite and accurate where
the spectrum vanishes. The Monte Carlo estimator writes the process as a
finite moving average of i.i.d. innovations using its **minimum-phase**
spectral factor, which makes $\{Y_t\}$ a hidden Markov process whose optimal
(fully adapted) particle filter is available in closed form; that filter runs
on one long typical sequence and Shannon–McMillan–Breiman turns its
log-likelihood into the rate.

## Contents

| | |
| --- | --- |
| [`paper/`](paper/) | the paper (`paper.tex`, built with `./build.sh paper`) |
| [`egp/`](egp/) | reference Python implementation — library + `egp` CLI, with CPU and WebGPU engines |
| [`experiments/`](experiments/) | the Section 5 experiments: one script per experiment, the results as JSON, and the scripts that turn them into the paper's figures and tables |
| [`webapp/`](webapp/) | interactive browser version: move the filter and the quantizer, watch the rate and real codecs respond |

## Quick start

```sh
# Python
pip install -e "egp[gpu]"          # drop [gpu] for the NumPy engine only
egp estimate --preset ma --taps 8 --delta 0.5 -n 20000 -N 1000 -r 3

# Paper
cd paper && ./build.sh paper

# Web app
cd webapp && npm install && npm run dev
```

## Reproducing the paper

Every figure and table in Section 5 is generated from the JSON results
committed under `experiments/results/`. To rebuild them from those results,
without rerunning anything:

```sh
cd experiments
python convergence_figures.py      # Figures 1-3, Table 1
python family_curves_figures.py    # Figures 4-8
python family_curves_tables.py     # the Appendix C tables
python compressors_figures.py      # Figure 9, Table 2
```

To regenerate the results themselves, see [`experiments/README.md`](experiments/README.md),
which gives the command line for each experiment together with the parameters
used for the runs reported in the paper. The estimator runs are seeded and
reproducible; the full set takes a few hours on a workstation GPU.

## Hosted build

A GitHub Actions workflow ([`.github/workflows/pages.yml`](.github/workflows/pages.yml))
builds both deliverables from source on every push to `main` and publishes
them to GitHub Pages: the interactive web app at the site root, and the
compiled paper at `/paper.pdf`. Neither is tracked in git, so what is
published is always built from the current sources.

Enabling it on a fresh clone is one manual step: in the repository's
**Settings → Pages**, set **Source** to **GitHub Actions**. The workflow can
also be run on demand from the Actions tab.

## Two implementations, one method

The TypeScript in `webapp/src/egp/` is a hand-synced port of the Python
package — change one, change the other. They are held together by golden
tests (`webapp/test/golden.test.ts`) that pin the port to values generated
from the Python original: the special functions to ~1e-12, the factorization
and spectral approximation to the precision each is worth, and a full
particle-filter run statistically.

The GPU path goes further and shares actual source: `egp/src/egp/pf.wgsl` is
one WGSL compute shader run by both hosts — through `wgpu-py` from Python, and
natively from the browser. On an integrated GPU it is roughly 20× the NumPy
filter.

## What the estimator will not do

Report a number it does not believe. When the quantization is fine relative to
the one-step prediction error, too small a particle cloud loses the state
entirely and the likelihood becomes arbitrary rather than noisy — and the
effective sample size does not notice, because the surviving particles agree
with each other whether or not they agree with the data. Every run therefore
carries the per-step diagnostic that does detect it; replicates that lost lock
are discarded rather than averaged, and a run with no survivors reports a
collapse instead of an estimate. The analytic spectral approximation, which
shares no code with the filter, sits beside every result as an independent
second opinion.
