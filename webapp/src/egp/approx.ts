/**
 * The analytic spectral approximation of Section 3, ported from egp/approx.py:
 * the classical high-resolution formula with the quantizer noise power Δ²/12
 * added to the spectrum, so spectral zeros stay finite.
 *
 * It is computed on a *midpoint* frequency grid — exact spectral nulls fall at
 * rational multiples of π, where log of a grid value would be −∞ even though
 * the integral converges — and needs no spectral floor, which makes it
 * independent of the factorization machinery: a genuinely external check on
 * the particle filter rather than a restatement of it.
 */
import { fft, nextPow2 } from './fft'

export const QUANTIZER_FLOOR_BITS = 0.5 * Math.log2((2 * Math.PI * Math.E) / 12)

/** |H(ω)|² on the midpoint grid ωₘ = π(m + ½)/n over [0, π]. */
export function psdMidpoint(kernel: Float64Array, n = 16384): Float64Array {
  const size = nextPow2(Math.max(2 * n, kernel.length))
  if (size !== 2 * n) throw new Error('psdMidpoint: n must be a power of two >= kernel length / 2')
  const re = new Float64Array(size)
  const im = new Float64Array(size)
  for (let j = 0; j < kernel.length; j++) {
    const phi = (-Math.PI / size) * j
    re[j] = kernel[j] * Math.cos(phi)
    im[j] = kernel[j] * Math.sin(phi)
  }
  fft(re, im)
  const spectrum = new Float64Array(n)
  for (let m = 0; m < n; m++) spectrum[m] = re[m] * re[m] + im[m] * im[m]
  return spectrum
}

/**
 * (1/4π) ∫ log₂( 2πe (S(ω) + Δ²/12) / Δ² ) dω, in bits/sample — the
 * closed-form reference of Section 3. Valid for Δ ≲ σ; in the coarse regime
 * it tends to the 0.2546-bit quantizer floor while the true rate tends to 0.
 */
export function entropyRateApprox(kernel: Float64Array, delta: number, n = 16384): number {
  const spectrum = psdMidpoint(kernel, n)
  const noise = (delta * delta) / 12
  let mean = 0
  for (let m = 0; m < n; m++) mean += Math.log2(spectrum[m] + noise)
  mean /= n
  return 0.5 * (Math.log2(2 * Math.PI * Math.E) + mean) - Math.log2(delta)
}

/** The part of `entropyRateApprox` above the fixed QUANTIZER_FLOOR_BITS. */
export function snrIntegral(kernel: Float64Array, delta: number, n = 16384): number {
  const spectrum = psdMidpoint(kernel, n)
  const noise = (delta * delta) / 12
  let mean = 0
  for (let m = 0; m < n; m++) mean += Math.log2(1 + spectrum[m] / noise)
  return (0.5 * mean) / n
}
