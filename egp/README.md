# egp — entropy rate of a quantized stationary Gaussian process

Estimates the entropy rate, in **bits per sample**, of

$$Y_t = \mathrm{round}(X_t / \Delta) \in \mathbb{Z},$$

where $X_t$ is a zero-mean stationary Gaussian process.  The method is the one
developed in the accompanying paper, [`../paper/paper.tex`](../paper/paper.tex): write the process
in FIR form $X_t = \sum_j h_j W_{t-j}$ with $W$ i.i.d. standard normal, draw one
long typical sequence, and evaluate its log-likelihood with a fully adapted
particle filter.  Shannon–McMillan–Breiman turns that into the entropy rate.

## Install

```bash
pip install -e .
```

Requires numpy, scipy and matplotlib.

## Specifying the process

Three equivalent entry points, all resolved internally to **minimum-phase** FIR
taps (which is essential — see "Why minimum phase" below):

| how | CLI | Python |
| --- | --- | --- |
| named preset | `--preset lowpass --cutoff 6000 --fs 30000 --taps 65` | `egp.from_preset(...)` |
| convolution kernel applied to i.i.d. N(0,1) | `--kernel taps.npy` | `egp.from_taps(...)` |
| autocovariance $\gamma_0,\gamma_1,\dots$ | `--autocov gamma.npy` | `egp.from_autocov(...)` |

Presets (`egp presets`): `white`, `ma`, `gaussian`, `lowpass`, `highpass`,
`bandpass`, `ar1`, `diff`.

The process is normalized to unit variance by default, so `--delta` is measured
in standard deviations; use `--sigma` or `--no-normalize` to change that.

## Estimating

```bash
egp estimate --preset ma --taps 8 --delta 0.5 -n 20000 -N 1000 -r 3
```

```
process     moving average (width 8)
taps L      8    sigma 1    h0 0.353522
delta       0.5    delta/sigma 0.5    delta/h0 1.414
factoring   autocov error 8.94e-09    floored bins 0.00%
filter      n 20000    particles 1000    repeats 3
  [1/3]   1.9201 bits/sample   mean   1.9201 +/-      --   ESS 79.3%   (  3.3s)
  [2/3]   1.9212 bits/sample   mean   1.9206 +/-  0.0005   ESS 79.3%   (  3.5s)
  [3/3]   1.9031 bits/sample   mean   1.9148 +/-  0.0058   ESS 79.6%   (  3.5s)

entropy rate   1.9148 +/- 0.0058 bits/sample
  bounds       1.5470  <=  est  <=  3.0620
  approx       1.9107 bits/sample  (-0.0041 vs estimate)
  order-0      3.0640 bits/sample (memoryless)
  per repeat   1.9201, 1.9212, 1.9031
  ESS          mean 79.4% of N, min 1.5%
  symbols      18    burn-in 0    elapsed 10.0 s
```

Each replicate is streamed to stderr as it finishes, with the running mean and
standard error, so a long run can be watched converge (and a losing one
abandoned early — a replicate that lost the state is tagged `LOST LOCK`).
stdout carries only the result, so `--json` stays machine-readable.

The `approx` line is the analytic approximation described below, computed for
free alongside the estimate as an independent reference — it shares no code
with the filter.

From Python the same stream is available through the `on_repeat` callback,
which receives a `RepeatUpdate` per replicate:

```python
egp.estimate_entropy_rate(spec, delta=0.5, n_repeats=5,
                           on_repeat=lambda u: print(u.index, u.running_mean_bits))
```

Knobs: `-n` sequence length, `-N` particles, `-r` replicates, `--seed`,
`--json FILE`, and `--burn-in` (leading steps filtered but excluded from the
rate; the default is 0 — see below).

```python
import egp

spec = egp.from_preset("lowpass", cutoff=6000, fs=30000, taps=33)
res = egp.estimate_entropy_rate(spec, delta=0.25, n=100_000, n_particles=50_000)
if not res.collapsed:
    print(res.entropy_rate_bits)
```

## GPU engine

The filter also runs on a GPU through WebGPU:

```bash
pip install -e ".[gpu]"     # adds wgpu
egp estimate --preset ma --taps 8 --delta 0.5 -n 20000 -N 1000 -r 3 --engine gpu
```

or from Python, `estimate_entropy_rate(spec, delta, engine="gpu")`;
`egp.gpu_available()` reports whether a device answers.  wgpu reaches any
Vulkan/Metal/D3D12 adapter, integrated GPUs included — no CUDA required.

The compute shader is `src/egp/pf.wgsl`, the **same WGSL source the
companion webapp runs in the browser** (imported there verbatim): one
workgroup of 256 threads runs one replicate, with the per-step reductions in
workgroup memory, and the repeats of a batch run as concurrent workgroups of
one dispatch.  Both engines generate identical observation sequences from the
same seeds, so they differ only in the filter's internal randomness: on the
example above an Intel Iris Xe reproduces the CPU numbers repeat for repeat
(1.9203/1.9210/1.9054 vs 1.9201/1.9212/1.9031) at 0.6 s per repeat instead of
3.5 s, and at N = 30 000 the speedup is ~20x.

The shader works in f32 (WebGPU has no f64), with log-space normal functions
rebuilt for that precision; weight errors are ~1e-4 relative — far below the
estimator's statistical resolution — and the per-step log-likelihoods are
accumulated in f64 on the CPU, where the lost-lock diagnostic runs exactly as
in the scalar filter.  `tests/test_pf_gpu.py` pins the two engines to each
other (skipped automatically when no adapter is present).

## Plotting

```bash
egp plot --preset lowpass --cutoff 6000 --fs 30000 --taps 65 --delta 0.5 -o fig.png
```

Six panels: sample path with the quantizer cells, symbol histogram against the
exact Gaussian marginal, power spectrum (target vs. realized), the convolution
kernel before and after factorization, the autocovariance, and a numeric
summary.  Nothing is estimated — this is for inspecting an example.

## Reading the output

Two rigorous bounds bracket the answer and are printed alongside it:

* **lower** $\;\bar h - \log_2\Delta\;$ with $\bar h=\tfrac12\log_2(2\pi e\,h_0^2)$,
  because $h(X_1^n) = H(Y_1^n) + h(X_1^n\mid Y_1^n) \le H(Y_1^n) + n\log\Delta$;
* **upper** $\;H(Y_1)$, the memoryless marginal entropy, since conditioning
  cannot increase entropy.

The estimate is biased **upward**, by Jensen's inequality applied to the
unbiased particle likelihood ($O(1/N)$) and by the finite-$n$ gap between
$H(Y_1^n)/n$ and its limit.  The convergence check is therefore to increase
`-N` and `-n` until the value stops decreasing.

`--burn-in` drops leading steps from the rate, which removes the transient in
which $H(Y_t \mid Y_1^{t-1})$ is still falling.  It defaults to 0 because that
transient is only $O(L/n)$ of the average and, measured on AR(1), is smaller
than the opposing drift from particle-cloud degradation — the two effects
cancel to within noise.  It is kept for studying the transient itself.

## How many particles?

The governing ratio is $\Delta/h_0$ — quantization step over one-step
prediction error — together with the state dimension $L-1$.  A smooth process
has $h_0 \ll \sigma$, meaning the observations pin down a state that lives in
many dimensions, and the particle cloud can lose it entirely.  When that
happens the filter does not return a noisy answer, it returns a meaningless
one.  For the 33-tap 6 kHz/30 kHz lowpass at $\Delta = 0.25\sigma$
($\Delta/h_0 = 6.1$, $L = 33$):

| N | estimate (bits/sample) | lost-lock steps | ESS |
| --: | --: | --: | --: |
| 300 | 2 728 000 | 94% | 25% |
| 10 000 | 2.1865 | 0% | 31% |
| 30 000 | 2.1755 | 0% | 31% |
| 120 000 | 2.1731 | 0% | 31% |

**Effective sample size does not detect this** — it is ~30% in every row,
because the surviving particles agree with each other whether or not they agree
with the data.  What does detect it is the per-step log-likelihood: a particle
$d$ prediction standard deviations from the observed cell contributes about
$\log(\Delta/h_0) - d^2/2$, so a filter that is tracking at all cannot fall far
below $\log\min(1, \Delta/h_0)$ nats, and steps ~25 nats under that are
lost-lock steps rather than unlucky symbols.  (The scale term matters: under
fine quantization *every* step is legitimately worth about $\log(\Delta/h_0)$
nats.)  `result.collapsed` combines that count with the bound check, and the
CLI refuses to print an estimate — exit code 1 — when it trips.

## Why minimum phase

Any convolution kernel with the right autocovariance defines the same process
law, but only the minimum-phase factor has a usable leading tap: $h_0$ is the
conditional standard deviation of $X_t$ given the past innovations, and it is
what the filter divides by.  A linear-phase design such as `firwin` output has a
negligible $h_0$ and would make the filter degenerate, so every input is passed
through a cepstral (Kolmogorov) factorization.

Two numerical settings matter there and are reported as `autocov error` and
`floored bins`:

* `--floor` regularizes spectral nulls before taking logs.  Setting it above a
  filter's true stopband power biases $h_0$ upward — for a 65-tap Hamming
  lowpass, `--floor 1e-8` overstates $h_0$ by 27%.  The default is `1e-14`.
* `--nfft` must be generous, because $\log S$ has integrable singularities
  wherever $S$ vanishes.  The default is at least $2^{18}$.

If `floored bins` is more than a fraction of a percent, $h_0$ partly reflects
`--floor` rather than the process, and the CLI says so.

## Performance

Cost is $O(nNL)$ with a Python-level loop over the $n$ steps, so short runs are
overhead-bound and long ones are dominated by the $N \times (L-1)$ matrix–vector
product. As a rough guide, `-n 100000 -N 2000` with $L = 8$ takes about a
minute per replicate.

## approx

```bash
egp approx --preset ma --taps 8 --delta 0.5
```

Evaluates $\tfrac{1}{4\pi}\int \log_2\big(2\pi e (S(\omega)+\Delta^2/12)/\Delta^2\big)\,d\omega$
from the spectrum with no sampling, split into a fixed
$\tfrac12\log_2(2\pi e/12) = 0.2546$ bit floor plus a spectral
signal-to-quantization-noise integral.  It costs milliseconds and agrees with
the particle filter to a few thousandths of a bit on the cases tested, so it is
the cheapest available check on an estimate.  Options: `--units {bits,nats}`,
`--grid`, `--json` (which also carries the entropy-power lower bound, the
classical fine-quantization value and the marginal bound).

## compress

```bash
egp compress --preset ma --taps 8 --delta 0.5 -n 200000
```

Generates a long realization and compresses it with every available codec on
each requested view of the stream, reporting bits per sample against a 16-bit
baseline.  The views are `raw`, `delta` (the first difference), and `lpc` (the
residual of a fixed-point linear predictor); each is integer-reversible, so
every reported size is that of a lossless code.  Options: `-n/--samples`,
`--lpc-order` (default $L-1$; 0 disables), `--transforms` (default `raw,lpc`),
`--methods`, `--estimate` (also run the particle filter, with `--estimate-n`,
`-N`, `-r`), `--seed`, `--json`, `-q/--quiet`.

`zlib`, `bz2` and `lzma` come from the standard library; `pip install -e
".[compress]"` adds zstd, brotli, lz4, ANS, and FLAC.  The FLAC entry is the
reference encoder at level 8 (`egp.flac_size`), driven through pyFLAC, or
through libsndfile if `soundfile` is installed instead; the two produce
byte-identical streams.
