# Expanding Section 5.3 (practical compressors vs. the entropy rate)

Notes on strengthening the codec comparison. Two threads: (A) test more codecs,
including real FLAC, and (B) add a model-aware coder that quantizes the
innovations of the minimum-phase representation, aiming to beat the generic LPC
pipeline.

## Current state (after Thread A)

- Reference rate: analytical approximation (validated in Section 5.2 for the
  step range used here).
- Coders reported: ANS on the raw stream (the order-0 reference); zlib-9,
  zstd-19, brotli-11, lzma-9, bz2-9 on the raw stream; delta coding followed by
  zlib or ANS; the reference FLAC encoder at level 8; and the package's
  fixed-point LPC (order 32) + ANS. Overheads above the rate at delta = sigma/4
  run 59% (zlib) to 3% (LPC + ANS), with FLAC at 16%.
- `compress_benchmark` takes `transforms=` (raw, delta, lpc), and
  `compressors.py` takes `-j` for parallel grid points.
- Web app still has its own TS codecs (DELTA_ZLIB, DELTA_ZSTD, DELTA_ANS, zstd
  via wasm) and has not been touched.

## Thread A: more codecs (done)

Goal: replace the single "general-purpose byte compressor" data point with a
small panel, and add the real FLAC codec so the audio-codec comparison is
against the actual standard rather than an emulation.

- [x] Report bz2-9 and lzma-9 (already in the registry) alongside zlib-9, on the
      raw stream. Cheap: just widen the `--methods` list and the table.
- [x] Add zstd-19 and brotli-11 (optional deps already wired) for a
      general-purpose panel.
- [x] Integrate real FLAC, as `egp.flac_size`: pyFLAC if installed, else
      libsndfile through soundfile, the two checked to be byte-identical. Note
      that the reference encoder at level 8 fits predictors only up to order 12,
      the streamable-subset limit at these sample rates; lifting that limit to 32
      (libFLAC's own maximum) changed nothing on any of the families, so its
      distance from the rate is not a matter of order. Running FLAC on the
      LPC(32) residual isolates where the rest of its gap sits: it is the Rice
      coding of the residuals, not the prediction.
- [x] Delta coding added as a transform stage (`delta_transform`), reported with
      zlib and with ANS.
- [x] Figure 9 keeps three families with the general-purpose panel drawn as a
      band; Figure 10 is new, the per-coder overhead at two steps; Table 2 lists
      the full panel at delta = sigma/4.

The sweep also got a `-j` flag, since the grid points are independent, and the
paper runs now give the expensive codecs only the stage they are reported on
(brotli alone was more than half the serial cost).

## Thread B: coding the innovations of the minimum-phase model

Idea: the minimum-phase inverse filter is exactly the whitening filter, so
coding the innovations is predictive coding with the model-matched predictor at
full order, rather than FLAC's data-fit order <= 32.

Key facts to keep straight:

- We only observe Y = round(X/Delta), not the continuous X, so the true iid
  innovations W_t are not recoverable exactly. The practical scheme must be an
  integer-reversible transform of the integer stream that approximates the
  whitening filter, then an entropy coder (ANS).
- Reversible transform => information-lossless => the scheme can approach the
  entropy rate. It cannot beat it.
- The whitening filter for an FIR-defined process is IIR (inverse of the FIR
  min-phase factor). An integer-reversible IIR filter needs a lifting/lattice
  implementation with attention to dynamic-range growth and stability.

Expected outcome:

- Largest gain over LPC(32) on the spectral-zero families (moving average, first
  difference), exactly where Section 5.3's footnote shows order 32 still leaves
  0.09 vs 0.32 bits on the table. On the smooth families LPC(32) is already near
  the floor, so expect little improvement.
- Plausible headline: closes much of the current 2-4% overhead toward ~1-2%,
  concentrated on MA/diff. Report honestly; do not expect to reach the rate.

Tasks:

- [ ] Derive the min-phase inverse (AR) coefficients from the factor `h`
      (`egp/src/egp/factor.py`), at an order long enough for the spectral-zero
      families.
- [ ] Implement an integer-reversible whitening transform (lifting) and its
      exact inverse; verify round-trip on integer streams.
- [ ] Entropy-code the whitened integers with ANS; account for the model header
      (O(L) coefficients, negligible at n = 1e6).
- [ ] Compare against LPC(32)+ANS across all seven families and the step range
      of Section 5.2. Add as a third coder in Figure 9 / Table 2.
- [ ] Sanity check: the model-aware coder should sit at or above the entropy
      rate at every point; a point below the rate signals a bug (the transform
      is not actually lossless, or the rate reference is off).

## Framing for the paper

- The thesis is unchanged: the compressibility of these signals lives in the
  correlations, and a coder must model that structure to approach the limit.
- Thread A shows that general-purpose coders (now a panel, not one point) all
  stay well above the rate, and that even the standard audio codec (real FLAC)
  leaves a measurable gap.
- Thread B shows that a coder built from the same minimum-phase model used for
  the estimator comes closest of all, which ties the compression section back to
  the machinery of Section 4.
