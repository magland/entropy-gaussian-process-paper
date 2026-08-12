/**
 * Seeded RNG for the particle filter: xoshiro128** uniforms (128-bit state,
 * period 2^128 − 1, seeded through splitmix32) with Box–Muller normals. One
 * replicate consumes ~n·N uniforms — easily 1e8 — so the 2^32-period
 * generators used elsewhere in the app are not enough here: their seeds are
 * mere offsets into one cyclic stream, and long refine-until-stopped runs
 * would reuse randomness across "independent" replicates.
 */
export class Rng {
  private s0: number
  private s1: number
  private s2: number
  private s3: number
  private spare: number | null = null

  constructor(seed: number) {
    // splitmix32 stream fills the state; any seed gives a non-zero state.
    let z = seed >>> 0
    const next = () => {
      z = (z + 0x9e3779b9) >>> 0
      let t = z
      t = Math.imul(t ^ (t >>> 16), 0x21f0aaad)
      t = Math.imul(t ^ (t >>> 15), 0x735a2d97)
      return (t ^ (t >>> 15)) >>> 0
    }
    this.s0 = next()
    this.s1 = next()
    this.s2 = next()
    this.s3 = next()
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1
  }

  /** Uniform on [0, 1). */
  uniform(): number {
    const s1 = this.s1
    const x = Math.imul(s1, 5)
    const result = (Math.imul((x << 7) | (x >>> 25), 9) >>> 0) / 4294967296
    const t = s1 << 9
    this.s2 ^= this.s0
    this.s3 ^= s1
    this.s1 = s1 ^ this.s2
    this.s0 ^= this.s3
    this.s2 ^= t
    this.s3 = (this.s3 << 11) | (this.s3 >>> 21)
    return result
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
