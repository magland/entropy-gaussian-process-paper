/**
 * Simulation of the FIR process and its uniform quantization — the port of
 * egp/process.py. Innovations W_{2−L} … W_n are drawn i.i.d. standard normal
 * so that every returned sample has its full history; there is no burn-in
 * transient, matching the "exact prior" initialization of the filter.
 */
import type { Rng } from './rng'

/** One stationary quantized draw y_t = round(x_t / delta), t = 1 … n. */
export function generateQuantized(taps: Float64Array, n: number, delta: number, rng: Rng): Int32Array {
  const L = taps.length
  const w = new Float64Array(n + L - 1)
  for (let i = 0; i < w.length; i++) w[i] = rng.normal()
  const y = new Int32Array(n)
  for (let t = 0; t < n; t++) {
    let x = 0
    // x_t = sum_j taps[j] * W_{t-j}; w is indexed so w[t + L - 1] = W_t.
    for (let j = 0; j < L; j++) x += taps[j] * w[t + L - 1 - j]
    y[t] = Math.floor(x / delta + 0.5)
  }
  return y
}

/** Derive an independent seed for replicate `index` from a base seed. */
export function replicateSeed(base: number, index: number): number {
  let z = (base ^ Math.imul(index + 1, 0x9e3779b9)) >>> 0
  z = Math.imul(z ^ (z >>> 16), 0x45d9f3b) >>> 0
  z = Math.imul(z ^ (z >>> 16), 0x45d9f3b) >>> 0
  return (z ^ (z >>> 16)) >>> 0
}
