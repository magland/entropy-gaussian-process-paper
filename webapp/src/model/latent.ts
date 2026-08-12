/**
 * A fixed latent randomness underlying everything: standard normals indexed
 * by absolute sample position. The pipeline x → h*x → round is evaluated on
 * demand against these indices, so changing σ or the filter transforms the
 * *same* underlying data — the display morphs smoothly instead of resampling
 * — and the compression block (indices 0…N) shares its randomness with the
 * displayed window.
 */
import { GaussianStream } from './random'

export const LATENT_SEED = 20260729

export class LatentSource {
  private xs: number[] = []
  private xStream: GaussianStream

  constructor(seed: number) {
    this.xStream = new GaussianStream(seed)
  }

  private ensure(n: number) {
    while (this.xs.length <= n) {
      this.xs.push(this.xStream.normal())
    }
  }

  /**
   * Quantized samples for absolute indices [start, start + count). The kernel
   * is applied zero-phase (centered on its midpoint), so changing its length
   * does not shift features along the time axis. Latent indices before 0 read
   * as zero input.
   */
  window(start: number, count: number, kernel: Float64Array, sigma: number): Int16Array {
    const L = kernel.length
    const mid = (L - 1) >> 1
    this.ensure(start + count - 1 + mid)
    const { xs } = this
    const out = new Int16Array(count)
    for (let j = 0; j < count; j++) {
      const n = start + j
      let y = 0
      for (let k = 0; k < L; k++) {
        const idx = n - k + mid
        if (idx >= 0) y += kernel[k] * xs[idx]
      }
      y *= sigma
      out[j] = Math.max(-32768, Math.min(32767, Math.round(y)))
    }
    return out
  }
}
