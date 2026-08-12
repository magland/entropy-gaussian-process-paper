/**
 * FIR filter presets and their frequency response.
 *
 * The filter is always realized as an explicit convolution kernel, so the
 * pipeline is exactly x → h*x → round, and the |H(f)| plotted is the response
 * of the same taps the data actually went through. Cutoffs are given in Hz
 * against a user-set sample rate; internally everything is in normalized
 * frequency (cycles/sample, Nyquist = 0.5).
 */

export type FilterFamily = 'none' | 'movingAverage' | 'lowpass' | 'bandpass' | 'firstDifference'

export interface FilterSpec {
  family: FilterFamily
  /** Low band edge, Hz (bandpass only). */
  lowHz: number
  /** Cutoff / high band edge, Hz (lowpass, bandpass). */
  highHz: number
  /** Kernel length for the windowed-sinc designs; forced odd. */
  taps: number
  /** Moving-average width. */
  width: number
}

export const FAMILY_LABELS: Record<FilterFamily, string> = {
  none: 'none (white noise)',
  movingAverage: 'moving average',
  lowpass: 'lowpass',
  bandpass: 'bandpass',
  firstDifference: 'first difference',
}

/** The default is the paper's running example — a moving average at Δ ≈ 0.5σ
 * — where the particle filter converges in seconds at the default N. Sharp
 * long filters (the bandpass at high σ) are where it visibly collapses, which
 * is a story best walked into rather than started in; the windowed-sinc
 * families therefore start at a short kernel, since L is both the hidden
 * state's dimension and a factor in the filter's O(nNL) cost. */
export const DEFAULT_SPEC: FilterSpec = {
  family: 'movingAverage',
  lowHz: 300,
  highHz: 6000,
  taps: 31,
  width: 8,
}

/**
 * Every control snaps to a ladder of round values — a slider that stops on
 * 6 kHz and 101 taps rather than 5847 Hz and 97. σ is dense through the
 * single digits where the interesting transitions live, then roughly ×1.5
 * steps; frequency adds 3, 4, 6, 8 so the usual band edges are reachable.
 */
export const SIGMA_STOPS = [0.5, 1, 2, 3, 4, 5, 8, 12, 20, 30, 50, 100]
export const TAP_STOPS = [9, 15, 21, 31, 45, 65, 101, 151, 201, 301]
export const WIDTH_STOPS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]

const FREQ_DECADE = [1, 1.5, 2, 3, 4, 5, 6, 8]

/** Round frequencies from 10 Hz up to `maxHz`. */
export function frequencyStops(maxHz: number): number[] {
  const out: number[] = []
  for (let decade = 10; decade <= 1e5; decade *= 10) {
    for (const m of FREQ_DECADE) {
      const v = m * decade
      if (v <= maxHz) out.push(v)
    }
  }
  return out
}

/** The stop nearest `v` (in log distance, so relative error is what counts). */
export function nearestStop(stops: number[], v: number): number {
  let best = stops[0]
  let bestErr = Infinity
  for (const s of stops) {
    const err = Math.abs(Math.log(s / v))
    if (err < bestErr) {
      bestErr = err
      best = s
    }
  }
  return best
}

/** The highest band edge the sample rate allows a stop to sit at. */
export function maxCutoffHz(sampleRateHz: number): number {
  return sampleRateHz * 0.49
}

/** Hamming-windowed sinc lowpass with unit DC gain; fc in cycles/sample. */
function windowedSincLowpass(fc: number, taps: number): Float64Array {
  const n = taps | 1
  const mid = (n - 1) / 2
  const h = new Float64Array(n)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const t = i - mid
    const sinc = t === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * t) / (Math.PI * t)
    const w = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1))
    h[i] = sinc * w
    sum += h[i]
  }
  for (let i = 0; i < n; i++) h[i] /= sum
  return h
}

/**
 * Snap the spec onto the control ladders and keep the band edges ordered and
 * below Nyquist — so what the sliders show is exactly what is designed.
 */
export function clampSpec(spec: FilterSpec, sampleRateHz: number): FilterSpec {
  const stops = frequencyStops(maxCutoffHz(sampleRateHz))
  const highHz = nearestStop(stops, spec.highHz)
  const below = stops.filter(f => f < highHz)
  return {
    ...spec,
    highHz,
    lowHz: below.length > 0 ? nearestStop(below, spec.lowHz) : highHz / 2,
    taps: nearestStop(TAP_STOPS, spec.taps),
    width: nearestStop(WIDTH_STOPS, spec.width),
  }
}

export function designKernel(spec: FilterSpec, sampleRateHz: number): Float64Array {
  const s = clampSpec(spec, sampleRateHz)
  switch (s.family) {
    case 'none':
      return new Float64Array([1])
    case 'movingAverage': {
      const w = Math.max(2, Math.round(s.width))
      return new Float64Array(w).fill(1 / w)
    }
    case 'lowpass':
      return windowedSincLowpass(s.highHz / sampleRateHz, s.taps)
    case 'bandpass': {
      const lo = windowedSincLowpass(s.lowHz / sampleRateHz, s.taps)
      const hi = windowedSincLowpass(s.highHz / sampleRateHz, s.taps)
      const h = new Float64Array(hi.length)
      for (let i = 0; i < h.length; i++) h[i] = hi[i] - lo[i]
      return h
    }
    case 'firstDifference':
      return new Float64Array([1, -1])
  }
}

/** ‖h‖₂ — the gain from input σ to the filtered signal's σ_y. */
export function kernelNorm(h: Float64Array): number {
  let sum = 0
  for (const v of h) sum += v * v
  return Math.sqrt(sum)
}

/** |H(f)| at `points` frequencies uniform on [0, 0.5] cycles/sample. */
export function magnitudeResponse(h: Float64Array, points: number): Float64Array {
  const out = new Float64Array(points)
  for (let k = 0; k < points; k++) {
    const f = (0.5 * k) / (points - 1)
    let re = 0
    let im = 0
    for (let i = 0; i < h.length; i++) {
      re += h[i] * Math.cos(2 * Math.PI * f * i)
      im -= h[i] * Math.sin(2 * Math.PI * f * i)
    }
    out[k] = Math.hypot(re, im)
  }
  return out
}
