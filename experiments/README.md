# Numerical experiments for the paper

Scripts reproducing the results of Section 5 of `../paper/paper.tex`.  Each
script runs one experiment over a parameter grid and writes a JSON file to
`results/`; the `*_figures.py` scripts turn those files into the figures and
tables in `figures/`.  Requires `egp` (`pip install -e ../egp`) and, for the
default GPU engine, `pip install -e "../egp[gpu]"`.

Scripts are named by content rather than by section number, since the
sections may be renumbered.  All default parameter values are modest, sized
for an integrated laptop GPU; every scale knob (grids, sequence length,
particle count, replicates) is a command-line flag, so the full-scale runs on
a larger machine use the same scripts with larger values.  Results are seeded
and reproducible; the observed sequences are identical across grid points
wherever that sharpens a comparison (see the docstring of each script).

Processes used (all normalized to sigma = 1, built in `common.py`):

| key | process | delta/h0 at delta = 0.25 |
| --- | --- | --- |
| `white` | white noise (exact, no filtering needed) | 0.25 |
| `diff` | first difference [1, -1] (spectral null at DC) | 0.35 |
| `ma` | moving average, width 8 | 0.71 |
| `ar1` | AR(1), rho = 0.9, 64 taps | 0.57 |
| `gaussian1` | Gaussian smoothing, tau = 1, 9 taps | 0.66 |
| `gaussian15` | Gaussian smoothing, tau = 1.5, 13 taps | 4.3 |
| `lowpass` | windowed-sinc lowpass, 6 kHz at fs = 30 kHz, 33 taps | 6.1 |

## Section 5.1 — convergence of the particle filter

```bash
python convergence_particles.py   # upward bias vs particle count N (ma, ar1, gaussian1)
python convergence_length.py      # estimate and scatter vs sequence length n (ma, ar1, gaussian1)
python convergence_collapse.py    # collapse diagnostic, fine quantization (lowpass, gaussian15)
python convergence_figures.py     # figures + markdown table from the JSON results
```

`convergence_particles.py` fixes n and sweeps N with common sequences,
isolating the O(1/N) bias of the filter as the paired excess over the largest
N.  `convergence_length.py` fixes N and sweeps n, showing the 1/sqrt(n)
shrinkage of the replicate scatter.  Both use seven processes spanning the
range of bias sizes at delta = 0.25 (moving averages of widths 8 to 64,
Gaussian kernels of widths 1 and 1.25, and the first difference); the AR(1)
process is omitted from the figures because its bias is unresolvably small
(results/convergence_particles_ar1.json).  The full-scale run is

```bash
python convergence_particles.py --particles 125 250 500 1000 2000 4000 8000 16000 128000 \
  -n 20000 -r 32
python convergence_length.py --lengths 2500 5000 10000 20000 40000 80000 160000 \
  -N 4000 --hard-particles 16000 -r 32 --hard-repeats 16
```

`convergence_collapse.py` sweeps N on the two strongly smoothing processes
(`lowpass` at delta = 0.25, `gaussian15` at delta = 0.5 so that its collapse
transition falls inside the default particle grid); the results record the
lost-lock diagnostic and the effective sample size, which fails to detect the
collapse.  The figure scripts merge every `<experiment>*.json` they find, so
a family added later can be run on its own into a suffixed results file.

## Section 5.2 — accuracy of the analytical approximation

```bash
python family_curves.py           # estimator + approximation over the family sweeps
python family_curves_figures.py   # one two-panel figure per family (rate + error)
python family_curves_tables.py    # appendix tables (LaTeX input + markdown)
```

`family_curves.py` runs both methods over each family: the approximation on
a dense parameter grid at four quantization steps (numerical integration of
the spectrum, no sampling), and the particle-filter estimator at about eight
parameter values per curve, so that the approximation is checked at every
part of every sweep.  White noise and the first difference have no family
parameter and are instead swept in the quantization step itself, out to the
coarse regime where the approximation fails.

Each point's particle count, sequence length, and replicate count are set
from its one-step prediction error `h0`, which governs how hard the
filtering problem is: above `--hard-h0` the base tier, below it the hard
tier, and below `--very-hard-h0` the very-hard tier.  Results are written
after every point, so an interrupted run keeps what it computed.  The
default parameters are a rough laptop pass; the full run raises them, e.g.

```bash
python family_curves.py -N 8000 -n 10000 -r 16 \
  --hard-particles 30000 --hard-samples 10000 --hard-repeats 16 \
  --very-hard-particles 130000 --very-hard-samples 5000 --very-hard-repeats 16
```

which takes about an hour on a workstation GPU.  Note the short sequences at
the hard tiers: since every step is an opportunity to lose lock while the
precision of the mean depends only on the total sample count `r * n`, more
and shorter replicates buy the same precision at a far smaller particle
count.

The figure and table scripts merge every `family_curves*.json` in
`results/`, later files overriding earlier ones point by point, so
individual points or families may be rerun at larger N into a suffixed
results file without repeating the whole sweep.  `family_curves_rerun.py` is such
a rerun, redoing the handful of lowpass points that lost a replicate at the
tier chosen for them.

```bash
python ma_wide_check.py           # is the wide-MA discrepancy filter bias?
```

`ma_wide_check.py` sweeps the particle count at a single point (moving
average of width 64 at delta = sigma/8), where the estimate lies above the
approximation.  Since both methods are biased upward, that can only be
residual filter bias, and the sweep measures how slowly it converges; this
is the one point of Section 5.2 where the comparison is limited by the
estimator rather than by the formula.

## Section 5.3 — practical compressors

```bash
python compressors.py             # zlib raw + LPC(32)/ANS across families and deltas
python compressors_figures.py     # figure + markdown table at delta = 0.25
```

CPU only, a few minutes.  Records both stages (raw and LPC residual) of both
codecs, the analytical approximation as the reference rate, and the empirical
order-0 entropy.  `--lpc-order` (default 32, the FLAC maximum) applies to
every family; order L - 1 is markedly worse for the spectral-zero families
(ma, diff).
