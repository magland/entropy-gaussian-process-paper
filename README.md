# Computing the entropy rate of a quantized stationary Gaussian process

This repository accompanies the paper of the same name. It contains the paper
itself, a reference Python implementation, an interactive browser
demonstration, and the scripts that reproduce every figure and table.

The compiled paper is published at
[magland.github.io/entropy-gaussian-process-paper/paper.pdf](https://magland.github.io/entropy-gaussian-process-paper/paper.pdf).
It can also be built locally with `cd paper && ./build.sh paper`.

## Contents

| | |
| --- | --- |
| [`paper/`](paper/) | the paper (`paper.tex`, built with `./build.sh paper`) |
| [`egp/`](egp/) | reference Python implementation: library and `egp` CLI, with CPU and WebGPU engines |
| [`experiments/`](experiments/) | the Section 5 experiments: one script per experiment, the results as JSON, and the scripts that turn them into the paper's figures and tables |
| [`webapp/`](webapp/) | interactive browser version: move the filter and the quantizer, watch the rate and real codecs respond |
| [`tutorials/`](tutorials/) | explanatory material, starting with a [tutorial on the particle filter](tutorials/particle-filter/) |

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
python compressors_figures.py      # Figures 9-10, Table 2
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
