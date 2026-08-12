/**
 * Iterative radix-2 complex FFT on split re/im Float64Arrays, in place.
 *
 * Everything the egp port needs — the cepstral factorization and the
 * midpoint-grid spectrum — runs on power-of-two grids of its own choosing, so
 * a plain radix-2 transform is enough. Sizes reach 2^18 for the
 * factorization, which is a few milliseconds.
 */

/** Smallest power of two >= n. */
export function nextPow2(n: number): number {
  let p = 1
  while (p < n) p *= 2
  return p
}

/** In-place FFT (inverse = true divides by N and conjugates the twiddles). */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length
  if (n !== im.length || (n & (n - 1)) !== 0) throw new Error('fft: length must be a power of two')

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      const tr = re[i]
      re[i] = re[j]
      re[j] = tr
      const ti = im[i]
      im[i] = im[j]
      im[j] = ti
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len
    const wRe = Math.cos(ang)
    const wIm = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cRe = 1
      let cIm = 0
      const half = len >> 1
      for (let k = 0; k < half; k++) {
        const a = i + k
        const b = a + half
        const uRe = re[a]
        const uIm = im[a]
        const vRe = re[b] * cRe - im[b] * cIm
        const vIm = re[b] * cIm + im[b] * cRe
        re[a] = uRe + vRe
        im[a] = uIm + vIm
        re[b] = uRe - vRe
        im[b] = uIm - vIm
        const nRe = cRe * wRe - cIm * wIm
        cIm = cRe * wIm + cIm * wRe
        cRe = nRe
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      re[i] /= n
      im[i] /= n
    }
  }
}
