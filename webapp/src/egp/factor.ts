/**
 * Spectral factorization: autocovariance ↔ minimum-phase FIR taps.
 *
 * Hand-synced TypeScript port of egp/factor.py — the cepstral (Kolmogorov)
 * construction of Section 4.2 of the paper: given an
 * autocovariance γ, find FIR taps h with Σⱼ hⱼ hⱼ₊ₖ = γₖ and all roots inside
 * the unit circle, so that h₀ equals the one-step prediction standard
 * deviation (Szegő).
 */
import { fft, nextPow2 } from './fft'

export interface Factorization {
  /** Minimum-phase taps, h[0] > 0. */
  taps: Float64Array
  /** One-step prediction standard deviation, taps[0]. */
  h0: number
  /** Fraction of spectrum grid points clipped by the floor — a warning sign
   * when more than a fraction of a percent. */
  flooredFraction: number
  /** max |γʰ − γ| / γ₀ over the target lags: the FIR truncation error. */
  autocovError: number
  nfft: number
}

/** γₖ = Σⱼ hⱼ hⱼ₊ₖ for k = 0 … maxLag. */
export function autocovarianceFromTaps(h: Float64Array, maxLag?: number): Float64Array {
  const n = h.length
  const K = Math.min(maxLag ?? n - 1, n - 1)
  const gamma = new Float64Array(K + 1)
  for (let k = 0; k <= K; k++) {
    let sum = 0
    for (let j = 0; j + k < n; j++) sum += h[j] * h[j + k]
    gamma[k] = sum
  }
  return gamma
}

/** Power spectrum on the full FFT grid: Sₘ = γ₀ + 2 Σₖ γₖ cos(ωₘ k). */
export function spectrumFromAutocov(gamma: Float64Array, nfft: number): Float64Array {
  const k = gamma.length - 1
  if (2 * k + 1 > nfft) throw new Error(`nfft=${nfft} too small for ${k + 1} autocovariance lags`)
  const re = new Float64Array(nfft)
  const im = new Float64Array(nfft)
  re.set(gamma)
  for (let j = 1; j <= k; j++) re[nfft - j] = gamma[j]
  fft(re, im)
  return re
}

/**
 * Minimum-phase spectral factor of an autocovariance sequence.
 *
 * The spectrum is floored at `floor · max(S)` before taking logs; log S has
 * integrable singularities wherever S vanishes, and only a generous grid
 * (nfft ≥ 2¹⁸ by default) keeps their contribution from being dominated by
 * the floor. Raising the floor above the true stopband power of a sharp
 * filter biases h₀ upward.
 */
export function minimumPhaseFromAutocov(
  gamma: Float64Array,
  nTaps: number,
  nfftWanted?: number,
  floor = 1e-14,
): Factorization {
  if (!(gamma[0] > 0)) throw new Error('gamma[0] must be positive')
  if (nTaps < 1) throw new Error('nTaps must be >= 1')
  const nfft = nextPow2(Math.max(nfftWanted ?? 0, 32 * nTaps, 32 * gamma.length, 1 << 18))

  const spec = spectrumFromAutocov(gamma, nfft)
  let smax = 0
  for (const v of spec) smax = Math.max(smax, v)
  if (!(smax > 0)) throw new Error('autocovariance has a non-positive spectrum')
  const thresh = floor * smax
  let floored = 0
  for (let i = 0; i < nfft; i++) {
    if (spec[i] < thresh) {
      spec[i] = thresh
      floored++
    }
  }

  // Real cepstrum of √S, folded to its causal part, then exponentiated.
  const re = new Float64Array(nfft)
  const im = new Float64Array(nfft)
  for (let i = 0; i < nfft; i++) re[i] = 0.5 * Math.log(spec[i])
  fft(re, im, true) // ceps = IFFT[½ ln S], real by symmetry
  const half = nfft / 2
  const causalRe = new Float64Array(nfft)
  causalRe[0] = re[0]
  for (let j = 1; j < half; j++) causalRe[j] = 2 * re[j]
  causalRe[half] = re[half]
  const causalIm = new Float64Array(nfft)
  fft(causalRe, causalIm)
  // exp of the complex spectrum, then back to time.
  for (let i = 0; i < nfft; i++) {
    const m = Math.exp(causalRe[i])
    causalRe[i] = m * Math.cos(causalIm[i])
    causalIm[i] = m * Math.sin(causalIm[i])
  }
  fft(causalRe, causalIm, true)

  let taps = causalRe.slice(0, nTaps)
  if (taps[0] < 0) for (let i = 0; i < taps.length; i++) taps[i] = -taps[i]
  for (const v of taps) {
    if (!Number.isFinite(v)) throw new Error('spectral factorization failed; try a larger floor or nfft')
  }
  if (!(taps[0] > 0)) throw new Error('spectral factorization failed; try a larger floor or nfft')

  const achieved = autocovarianceFromTaps(taps, gamma.length - 1)
  let maxErr = 0
  for (let k = 0; k < gamma.length; k++) {
    const a = k < achieved.length ? achieved[k] : 0
    maxErr = Math.max(maxErr, Math.abs(a - gamma[k]))
  }
  return {
    taps,
    h0: taps[0],
    flooredFraction: floored / nfft,
    autocovError: maxErr / gamma[0],
    nfft,
  }
}

/**
 * Minimum-phase kernel with the same autocovariance as `kernel`.
 *
 * Any convolution kernel defines the same Gaussian process law as its
 * minimum-phase counterpart, but only the minimum-phase version has a usable
 * leading tap — a linear-phase design's leading tap is tiny, which would make
 * the particle filter degenerate.
 */
export function minimumPhaseFromTaps(
  kernel: Float64Array,
  nTaps?: number,
  nfft?: number,
  floor = 1e-14,
): Factorization {
  if (kernel.length === 0) throw new Error('kernel is empty')
  const gamma = autocovarianceFromTaps(kernel)
  return minimumPhaseFromAutocov(gamma, nTaps ?? kernel.length, nfft, floor)
}
