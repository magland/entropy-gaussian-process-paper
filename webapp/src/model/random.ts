/**
 * Deterministic Gaussian sample stream.
 *
 * Seeded so that the compression block is reproducible for a given parameter
 * set — the reported sizes then only move when the model moves, not between
 * recomputes. mulberry32 for the uniforms, Box–Muller for the normals.
 */
export class GaussianStream {
  private state: number
  private spare: number | null = null

  constructor(seed: number) {
    this.state = seed >>> 0
  }

  private uniform(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0
    let t = this.state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  /** Standard normal. */
  normal(): number {
    if (this.spare !== null) {
      const v = this.spare
      this.spare = null
      return v
    }
    let u = 0
    while (u === 0) u = this.uniform()
    const r = Math.sqrt(-2 * Math.log(u))
    const theta = 2 * Math.PI * this.uniform()
    this.spare = r * Math.sin(theta)
    return r * Math.cos(theta)
  }
}
