/**
 * The two closed-form bounds noted in Section 5.1, ported from egp/estimate.py:
 *
 *     h̄ − log₂Δ  ≤  R̄  ≤  H(Y₁),      h̄ = ½ log₂(2πe h₀²).
 *
 * They are too loose to be worth a reader's attention — the spectral
 * approximation is a far sharper reference — so nothing in the UI shows them.
 * They stay here because they are exact identities in the white-noise and
 * fine-quantization limits, which makes them useful for validating the port
 * against the Python package (see test/golden.test.ts).
 */
import { logCellProb } from './pf'

const LOG2 = Math.log(2)

/** Differential entropy of N(0, scale²) in bits. */
export function differentialEntropyBits(scale: number): number {
  return 0.5 * Math.log2(2 * Math.PI * Math.E * scale * scale)
}

/** Fine-quantization value h̄ − log₂Δ, a rigorous lower bound on R̄. */
export function fineQuantizationBits(h0: number, delta: number): number {
  return differentialEntropyBits(h0) - Math.log2(delta)
}

/**
 * H(Y₁) in bits for a single N(0, σ²) sample rounded to a multiple of Δ.
 * Because conditioning cannot increase entropy this is a rigorous upper bound
 * on the entropy rate of any quantized stationary process with this marginal.
 */
export function marginalEntropyBits(sigma: number, delta: number, maxSymbols = 4_000_000): number {
  const reach = Math.ceil((10 * sigma) / delta) + 1
  if (2 * reach + 1 > maxSymbols) {
    // Too fine to enumerate; the fine-quantization asymptote is exact here.
    return differentialEntropyBits(sigma) - Math.log2(delta)
  }
  let entropy = 0
  for (let y = -reach; y <= reach; y++) {
    const lo = ((y - 0.5) * delta) / sigma
    const logP = logCellProb(lo, lo + delta / sigma)
    const p = Math.exp(logP)
    if (p > 0) entropy -= p * logP
  }
  return entropy / LOG2
}
