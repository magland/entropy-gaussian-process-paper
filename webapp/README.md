# webapp — entropy rate of a quantized Gaussian process, in the browser

Interactive companion to the paper ([`../paper/paper.tex`](../paper/paper.tex))
and the [`../egp`](../egp) Python package, built entirely on this project's
methodology.

The generating model is: i.i.d. Gaussian noise (std σ, measured in
quantization steps) → FIR filter → round to integers. The app shows the
filter (design kernel, its **minimum-phase spectral factor**, and the
frequency response), a window of the generated integer signal (stationary by
default, with a play toggle), and the measured compression of a block of the
generated integers under the Section 5.3 benchmark — zlib, zstd, and an rANS
entropy coder, each raw, delta-coded, and LPC-residual-coded — as bits per
sample and as ratio against raw int16 storage.

Everything the paper says to compute alongside an estimate is on screen:

- **The entropy rate R̄** — the bits/sample limit no lossless code can beat —
  is estimated in the browser by the method of the paper:
  Shannon–McMillan–Breiman applied to the likelihood of one long typical
  sequence, evaluated by the **fully adapted particle filter** of Section 4.3 on
  the minimum-phase FIR form of the process (Section 4.2, cepstral
  construction). A button starts a web worker that streams independent
  replicates (live mean ± se, dashed line on the chart) until stopped. Two
  engines: scalar TypeScript, and **WebGPU** (the default where available) —
  each replicate is one workgroup, 256 threads across the particles with the
  per-step max/scan reductions in workgroup memory, and several replicates
  run as concurrent workgroups of one dispatch. The WGSL side rebuilds the
  log-space normal functions for f32 (A&S erf + Laplace continued fraction
  for log Φ; Acklam-seeded Newton for Φ⁻¹(eʸ), with the tails computed from
  log p directly where exp would cancel); per-step log-likelihoods are read
  back and accumulated in f64, where the same lost-lock diagnostic runs. The
  shader source is `../egp/src/egp/pf.wgsl`, shared verbatim with the
  Python package's GPU engine (`egp estimate --engine gpu`).
- **The analytic spectral approximation** of Section 3 — the high-resolution
  formula with the quantizer's Δ²/12 noise floor filling the spectral zeros —
  is the dotted reference line, recomputed instantly on every parameter
  change. It shares no code with the filter, which makes it an external check
  rather than a restatement, and it is sharp enough to be the only reference
  worth showing. (The closed-form bounds noted in Section 5.1 are implemented in
  `src/egp/bounds.ts` and used by the tests, but they are too loose to help a
  reader, so nothing in the UI reports them.)
- **Collapse diagnostics.** The per-step lost-lock counter — the diagnostic
  that works — is evaluated on every run; a collapsed estimate is flagged in
  the UI and kept off the chart, because it is meaningless rather than noisy,
  and the ESS readout is shown precisely so you can watch it *fail* to notice.
  The Δ/h₀ and σ/h₀ readouts warn when the model is in the many-particles
  regime before you press the button.

The default model is the paper's running example (MA(8) at Δ ≈ 0.5σ), where
N = 1000 particles converge in a few seconds per replicate. Push toward the
sharp bandpass at high σ and the collapse story of Section 5.1 and Appendix B plays out live.

## Run it

```sh
npm install
npm run dev
```

## Validation

`src/egp/` is a hand-synced TypeScript port of the `../egp` package — change
one, change the other. The port is pinned to the original by golden tests:

```sh
npm test               # vitest against golden.json
npm run golden         # regenerate golden.json (needs pip install -e ../egp)
```

The tests check the special functions (`log_ndtr`, `ndtri_exp`, the log-space
cell probability) to ~1e-12 against scipy, the factorization, approximation
and closed-form bounds against egp (with documented looser tolerances for
sharp filters,
where the deep stopband is cancellation noise at the spectral floor and the
factorization is ill-conditioned — the autocovariance of the factor, i.e. the
process law, still matches to ~1e-10), the exact white-noise identity, and a
full particle-filter run against the Python reference statistically. The
in-app copyable command reruns the Python original at the current settings as
an independent check.

## Layout

```
src/model/       the latent source (fixed seeded randomness indexed by sample
                 position, convolved zero-phase with the kernel on demand)
                 and the FIR presets
src/egp/        the hand-synced port of ../egp: cepstral minimum-phase
                 factorization (factor.ts, fft.ts), the fully adapted particle
                 filter in log space (pf.ts, normal.ts, rng.ts) and its WebGPU
                 twin (pfGpu.ts), the spectral approximation (approx.ts), the
                 closed-form bounds used by the tests (bounds.ts), and process
                 generation (process.ts)
src/compress/    lossless codecs run in the browser: zlib (fflate), zstd
                 (wasm), ans.ts (a bit-identical port of simple_ans), and
                 FLAC-style integer LPC
src/worker/      the model analysis (factorization + bounds + approximation),
                 the estimator (one replicate at a time until terminated), and
                 the codecs, all off the main thread
src/components/  controls, filter plots (including the minimum-phase panel),
                 signal canvas, compression chart with the reference lines and
                 bounds band, and the method note
```

Every reported size round-trips through the decoder and includes whatever the
decoder needs (ANS symbol table, LPC coefficients). The signal view and the
compression block read the same fixed latent noise sequence — parameter
changes transform the same underlying data rather than resampling it. The
entropy estimator draws its own fresh sequences per replicate (as the
methodology requires for honest error bars); both are draws from the same
process law, since the minimum-phase factor is law-equivalent to the design
kernel.
