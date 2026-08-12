/**
 * Fully adapted particle filter for the log-likelihood of a quantized sample.
 *
 * Hand-synced TypeScript port of egp/pf.py, implementing Section 4.3 of the
 * paper. The hidden state is the window of the last L−1
 * innovations; given that window, X_t is N(μ, h₀²), so the observation weight
 * is the Gaussian cell probability Φ(r) − Φ(l) and the optimal proposal for
 * the new innovation is N(0, 1) truncated to [l, r). Both are available in
 * closed form, which makes the filter fully adapted: resampling uses the
 * exact predictive weights and no importance weights survive the step.
 *
 * All internal quantities are in nats; callers convert to bits.
 */
import { logNdtr, ndtriExp } from './normal'
import type { Rng } from './rng'

const LOG2 = Math.log(2)

/** log(1 − eˣ) for x ≤ 0, accurate at both ends. */
export function log1mexp(x: number): number {
  return x > -LOG2 ? Math.log(-Math.expm1(x)) : Math.log1p(-Math.exp(x))
}

function logaddexp(a: number, b: number): number {
  if (a === -Infinity) return b
  if (b === -Infinity) return a
  const m = Math.max(a, b)
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}

/**
 * log(Φ(hi) − Φ(lo)) for hi ≥ lo, stable deep into either tail.
 *
 * Reflecting an interval whose midpoint is positive keeps the pair of
 * endpoints on the side where logNdtr retains relative precision; without it,
 * two large positive endpoints both give logNdtr ≈ 0 and their difference
 * underflows.
 */
export function logCellProb(lo: number, hi: number): number {
  const mirror = lo + hi > 0
  const a = mirror ? -hi : lo
  const b = mirror ? -lo : hi
  const la = logNdtr(a)
  const lb = logNdtr(b)
  return lb + log1mexp(la - lb)
}

export const DEFAULT_COLLAPSE_MARGIN = 25

/**
 * Fraction of lost-lock steps above which a replicate is unusable. A filter
 * that is tracking has none at all, so this is a wide margin around zero
 * rather than a tuned cutoff. Such a replicate's likelihood is an arbitrary
 * number, not a noisy one, so it must be discarded rather than averaged in.
 */
export const LOST_LOCK_TOLERANCE = 1e-3

/**
 * Per-step log-likelihood below which the filter has certainly lost the state.
 *
 * A particle sitting d prediction standard deviations from the observed cell
 * contributes about log(Δ/h₀) − d²/2, so a filter that is tracking at all
 * cannot fall far below log(min(1, Δ/h₀)). Anything `margin` nats under that
 * is a lost-lock step, not an unlucky symbol. The scale term matters: under
 * fine quantization *every* step is legitimately worth about log(Δ/h₀) nats,
 * so a fixed threshold would misfire. (ESS does not detect this failure —
 * the surviving particles agree with each other whether or not they agree
 * with the data.)
 */
export function collapseThreshold(delta: number, h0: number, margin = DEFAULT_COLLAPSE_MARGIN): number {
  return Math.log(Math.min(1, delta / h0)) - margin
}

export interface FilterResult {
  loglikNats: number
  nSteps: number
  /** −loglik / (n ln 2): the entropy-rate estimate in bits per sample. */
  entropyRateBits: number
  /** Effective sample size, averaged over steps, as a fraction of N. */
  meanEss: number
  minEss: number
  nLostLock: number
  /** −1 if the filter never lost lock. */
  firstLostStep: number
  lostLockFraction: number
  collapseThreshold: number
}

/**
 * Estimate log P(y₁…yₙ) for the quantized FIR process.
 *
 * `taps` must be the minimum-phase factorization with taps[0] > 0;
 * `y[t] = round(x[t] / delta)`. For a white process (L = 1) the likelihood is
 * exact and `nParticles` is ignored.
 */
export function particleFilterLoglik(
  y: Int32Array,
  taps: Float64Array,
  delta: number,
  nParticles: number,
  rng: Rng,
  collapseMargin = DEFAULT_COLLAPSE_MARGIN,
  progress?: (done: number, total: number) => void,
): FilterResult {
  const n = y.length
  const nTaps = taps.length
  const h0 = taps[0]
  if (!(h0 > 0)) throw new Error('taps[0] must be positive (use the minimum-phase factorization)')
  if (!(delta > 0)) throw new Error('delta must be positive')
  const limit = collapseThreshold(delta, h0, collapseMargin)
  const width = delta / h0

  const finish = (loglik: number, meanEss: number, minEss: number, nLost: number, firstLost: number): FilterResult => ({
    loglikNats: loglik,
    nSteps: n,
    entropyRateBits: n > 0 ? -loglik / (n * LOG2) : NaN,
    meanEss,
    minEss,
    nLostLock: nLost,
    firstLostStep: firstLost,
    lostLockFraction: n > 0 ? nLost / n : 0,
    collapseThreshold: limit,
  })

  if (n === 0) return finish(0, 1, 1, 0, -1)

  if (nTaps === 1) {
    // White: symbols are i.i.d., the likelihood is exact.
    let loglik = 0
    let nLost = 0
    let firstLost = -1
    for (let t = 0; t < n; t++) {
      const lo = ((y[t] - 0.5) * delta) / h0
      const step = logCellProb(lo, lo + width)
      loglik += step
      if (step < limit) {
        nLost++
        if (firstLost < 0) firstLost = t
      }
    }
    return finish(loglik, 1, 1, nLost, firstLost)
  }

  const N = nParticles | 0
  if (N < 2) throw new Error('nParticles must be >= 2')

  const ringLen = nTaps - 1
  // Particle-major ring buffer of the last L−1 innovations. The tap vector is
  // rolled each step (tapAt below) so that μ is a plain contiguous dot product
  // per particle — the O(nNL) inner loop of the whole estimator.
  let ring = new Float64Array(N * ringLen)
  for (let i = 0; i < ring.length; i++) ring[i] = rng.normal()
  let scratch = new Float64Array(N * ringLen)
  const tapAt = new Float64Array(ringLen)

  const lo = new Float64Array(N)
  const logAlpha = new Float64Array(N)
  const logLower = new Float64Array(N)
  const mirror = new Uint8Array(N)
  const cumulative = new Float64Array(N)
  const ancestors = new Int32Array(N)

  let pos = 0
  let loglik = 0
  let essSum = 0
  let essMin = N
  let nLost = 0
  let firstLost = -1
  const logN = Math.log(N)
  const reportEvery = Math.max(1, Math.floor(n / 100))

  for (let t = 0; t < n; t++) {
    // Rolled tap vector so ring[·, (pos − k) mod ringLen] pairs with taps[1+k].
    for (let k = 0; k < ringLen; k++) {
      tapAt[(((pos - k) % ringLen) + ringLen) % ringLen] = taps[1 + k]
    }
    const lower = (y[t] - 0.5) * delta

    // Weights: the Gaussian cell probability per particle, in log space. The
    // weight and the optimal proposal share all their transcendentals.
    let peak = -Infinity
    for (let i = 0; i < N; i++) {
      let mu = 0
      const base = i * ringLen
      for (let j = 0; j < ringLen; j++) mu += ring[base + j] * tapAt[j]
      const l = (lower - mu) / h0
      const h = l + width
      const flip = l + h > 0
      const a = flip ? -h : l
      const b = flip ? -l : h
      const la = logNdtr(a)
      const lb = logNdtr(b)
      lo[i] = l
      logAlpha[i] = lb + log1mexp(la - lb)
      logLower[i] = la
      mirror[i] = flip ? 1 : 0
      if (logAlpha[i] > peak) peak = logAlpha[i]
    }
    if (!Number.isFinite(peak)) {
      throw new Error(
        `particle filter collapsed at step ${t}: every particle assigns zero ` +
          'probability to the observed symbol. Increase the particle count.',
      )
    }

    let wSum = 0
    let wSumSq = 0
    for (let i = 0; i < N; i++) {
      const w = Math.exp(logAlpha[i] - peak)
      wSum += w
      cumulative[i] = wSum
      wSumSq += w * w
    }
    const step = peak + Math.log(wSum) - logN
    loglik += step
    if (step < limit) {
      nLost++
      if (firstLost < 0) firstLost = t
    }
    const ess = (wSum * wSum) / wSumSq
    essSum += ess
    if (ess < essMin) essMin = ess

    // Systematic resampling with the exact predictive weights; the stratified
    // u's are increasing, so one merge pointer replaces per-particle search.
    const u0 = rng.uniform()
    let j = 0
    for (let i = 0; i < N; i++) {
      const u = ((u0 + i) / N) * wSum
      while (j < N - 1 && cumulative[j] < u) j++
      ancestors[i] = j
    }
    for (let i = 0; i < N; i++) {
      const src = ancestors[i] * ringLen
      const dst = i * ringLen
      for (let k = 0; k < ringLen; k++) scratch[dst + k] = ring[src + k]
    }
    const prev = ring
    ring = scratch
    scratch = prev

    // Propagate: the optimal proposal is the truncated innovation, drawn by
    // inverse CDF in log space from the ancestor's precomputed cell stats.
    pos = (pos + 1) % ringLen
    for (let i = 0; i < N; i++) {
      const a = ancestors[i]
      const u = rng.uniform()
      const logP = logaddexp(logLower[a], Math.log(u) + logAlpha[a])
      let x = ndtriExp(logP)
      if (mirror[a]) x = -x
      const l = lo[a]
      const h = l + width
      if (!Number.isFinite(x)) x = 0.5 * (l + h) // numerically empty cell
      ring[i * ringLen + pos] = x < l ? l : x > h ? h : x
    }

    if (progress && (t % reportEvery === 0 || t === n - 1)) progress(t + 1, n)
  }

  return finish(loglik, essSum / (n * N), essMin / N, nLost, firstLost)
}
