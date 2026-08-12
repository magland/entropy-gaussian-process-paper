/**
 * Linear predictive coding, the generalisation of delta coding.
 *
 * Delta codes `x[n] - x[n-1]`. LPC fits an order-p predictor
 *
 *     xhat[n] = sum_{k=1..p} a_k x[n-k]
 *
 * to the data and codes the residual `x[n] - xhat[n]` instead. Delta is the
 * special case p = 1, a_1 = 1.
 *
 * For the residual to be *losslessly* invertible the decoder has to reproduce
 * `xhat` bit-for-bit, so — as in FLAC and Shorten — the coefficients are
 * quantised to integers with a shared right shift and the prediction is done
 * in exact integer arithmetic:
 *
 *     xhat[n] = floor( (sum_k q_k x[n-k]) / 2^shift )
 *
 * `sum_k q_k x[n-k]` reaches about 2^35 for the orders and widths used here,
 * which is exact in a double, so no BigInt is needed.
 */

/** 15-bit signed coefficients, as FLAC's default. */
const COEFF_PRECISION = 15

export interface LpcModel {
  order: number
  shift: number
  coeffs: Int32Array
}

/** Autocorrelation r[0..maxLag] of the signal. */
function autocorrelation(x: Int16Array, maxLag: number): Float64Array {
  const r = new Float64Array(maxLag + 1)
  for (let lag = 0; lag <= maxLag; lag++) {
    let sum = 0
    for (let i = lag; i < x.length; i++) sum += x[i] * x[i - lag]
    r[lag] = sum
  }
  return r
}

/**
 * Levinson-Durbin: the order-p predictor minimising the mean squared
 * prediction error, from the autocorrelation. Returns a[1..p] in a[0..p-1].
 */
function levinson(r: Float64Array, order: number): Float64Array | null {
  if (r[0] <= 0) return null
  const a = new Float64Array(order)
  const tmp = new Float64Array(order)
  let error = r[0]

  for (let i = 0; i < order; i++) {
    let acc = r[i + 1]
    for (let j = 0; j < i; j++) acc -= a[j] * r[i - j]
    const k = acc / error
    if (!Number.isFinite(k)) return null

    a[i] = k
    for (let j = 0; j < i; j++) tmp[j] = a[j] - k * a[i - 1 - j]
    for (let j = 0; j < i; j++) a[j] = tmp[j]

    error *= 1 - k * k
    if (error <= 0) return null
  }
  return a
}

/** Scale the real coefficients onto integers sharing one right shift. */
function quantiseCoefficients(a: Float64Array): LpcModel | null {
  let maxAbs = 0
  for (const v of a) maxAbs = Math.max(maxAbs, Math.abs(v))
  if (maxAbs === 0 || !Number.isFinite(maxAbs)) return null

  let shift = COEFF_PRECISION - 1 - Math.floor(Math.log2(maxAbs)) - 1
  shift = Math.max(0, Math.min(15, shift))

  const limit = 2 ** (COEFF_PRECISION - 1)
  const coeffs = new Int32Array(a.length)
  for (let i = 0; i < a.length; i++) {
    coeffs[i] = Math.max(-limit, Math.min(limit - 1, Math.round(a[i] * 2 ** shift)))
  }
  return { order: a.length, shift, coeffs }
}

export function fitLpc(x: Int16Array, order: number): LpcModel | null {
  const a = levinson(autocorrelation(x, order), order)
  return a ? quantiseCoefficients(a) : null
}

/** The unquantised order-p predictor, for coders that keep full precision. */
export function fitRealCoeffs(x: Int16Array, order: number): Float64Array | null {
  return levinson(autocorrelation(x, order), order)
}

/**
 * Prediction residual. The first `order` entries are the samples themselves
 * (the warm-up the decoder needs before it can predict). Differences are taken
 * in int16 with wraparound, so the transform is exactly invertible.
 */
export function lpcResidual(x: Int16Array, model: LpcModel): Int16Array {
  const { order, shift, coeffs } = model
  const scale = 2 ** shift
  const e = new Int16Array(x.length)
  for (let n = 0; n < Math.min(order, x.length); n++) e[n] = x[n]
  for (let n = order; n < x.length; n++) {
    let sum = 0
    for (let k = 0; k < order; k++) sum += coeffs[k] * x[n - 1 - k]
    e[n] = ((x[n] - Math.floor(sum / scale)) << 16) >> 16
  }
  return e
}

/** Rebuild the signal from its residual — the exact inverse of `lpcResidual`. */
export function lpcRestore(e: Int16Array, model: LpcModel): Int16Array {
  const { order, shift, coeffs } = model
  const scale = 2 ** shift
  const x = new Int16Array(e.length)
  for (let n = 0; n < Math.min(order, e.length); n++) x[n] = e[n]
  for (let n = order; n < e.length; n++) {
    let sum = 0
    for (let k = 0; k < order; k++) sum += coeffs[k] * x[n - 1 - k]
    x[n] = ((e[n] + Math.floor(sum / scale)) << 16) >> 16
  }
  return x
}

/** Bytes the decoder needs for the model: order, shift, and the coefficients. */
export function modelSize(model: LpcModel): number {
  return 2 + 2 * model.order
}
