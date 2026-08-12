/**
 * Everything derivable from the model without sampling, off the main thread:
 * the minimum-phase factorization (four FFTs on a 2^18 grid — too slow for a
 * slider's render path) and the spectral approximation. One message in
 * (design kernel and σ), one message out.
 *
 * The pipeline quantizes to integers, so throughout Δ = 1 and the process
 * taps are σ · kernel: Δ/σ on the paper's normalized scale is 1/σ_Y here.
 */
import { minimumPhaseFromTaps } from '../egp/factor'
import { entropyRateApprox } from '../egp/approx'

export interface ModelRequest {
  id: number
  kernel: Float64Array
  sigma: number
}

export interface ModelAnalysis {
  /** Minimum-phase taps of the scaled process (σ · kernel), for the filter. */
  minPhaseTaps: Float64Array
  /** One-step prediction standard deviation σ·h̃₀ (Szegő); Δ = 1. */
  h0: number
  /** Marginal standard deviation of the pre-quantization signal. */
  sigmaY: number
  /** Factorization diagnostics — the FIR truncation check of Section 4.2. */
  autocovError: number
  flooredFraction: number
  /** The Section 3 spectral approximation, bits/sample. */
  approxBits: number
}

export interface ModelResponse {
  id: number
  analysis?: ModelAnalysis
  error?: string
}

const post = self.postMessage as (message: ModelResponse) => void

self.onmessage = (e: MessageEvent<ModelRequest>) => {
  const { id, kernel, sigma } = e.data
  try {
    const taps = new Float64Array(kernel.length)
    for (let i = 0; i < kernel.length; i++) taps[i] = sigma * kernel[i]
    const fac = minimumPhaseFromTaps(taps)
    let energy = 0
    for (const v of taps) energy += v * v
    const sigmaY = Math.sqrt(energy)
    post({
      id,
      analysis: {
        minPhaseTaps: fac.taps,
        h0: fac.h0,
        sigmaY,
        autocovError: fac.autocovError,
        flooredFraction: fac.flooredFraction,
        approxBits: entropyRateApprox(taps, 1),
      },
    })
  } catch (err) {
    post({ id, error: String(err) })
  }
}
