/**
 * Standard normal special functions for the particle filter, standing in for
 * scipy.special.{ndtr, ndtri, log_ndtr, ndtri_exp} in the Python package.
 *
 * Φ comes from W. J. Cody's rational erfc (SPECFUN calerf), full double
 * precision with correct relative accuracy deep in the lower tail. Φ⁻¹ is
 * Acklam's approximation polished by one Halley step against this Φ.
 *
 * The two log-space entry points the filter actually uses are logNdtr and
 * ndtriExp. logNdtr is log Φ evaluated with full relative accuracy over the
 * whole line: through erfc where it is representable, and by the standard
 * Mills-ratio asymptotic series below x ≈ -37 where erfc underflows.
 * ndtriExp inverts it — Φ⁻¹(eʸ) for y as small as the filter's log cell
 * masses get — by Newton iteration on log Φ, whose update
 * x ← x − (log Φ(x) − y)·Φ(x)/φ(x) never overflows.
 */

const THRESH = 0.46875
const SQRPI = 5.6418958354775628695e-1 // 1/√π

const ERF_A = [3.1611237438705656, 1.13864154151050156e2, 3.77485237685302021e2, 3.20937758913846947e3, 1.85777706184603153e-1]
const ERF_B = [2.36012909523441209e1, 2.44024637934444173e2, 1.28261652607737228e3, 2.84423683343917062e3]
const ERF_C = [5.64188496988670089e-1, 8.88314979438837594, 6.61191906371416295e1, 2.98635138197400131e2, 8.8195222124176909e2, 1.71204761263407058e3, 2.05107837782607147e3, 1.23033935479799725e3, 2.15311535474403846e-8]
const ERF_D = [1.57449261107098347e1, 1.17693950891312499e2, 5.37181101862009858e2, 1.62138957456669019e3, 3.29079923573345963e3, 4.36261909014324716e3, 3.43936767414372164e3, 1.23033935480374942e3]
const ERF_P = [3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2]
const ERF_Q = [2.56852019228982242, 1.87295284992346047, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3]

/** erf(x) for |x| ≤ 0.46875. */
function erfSmall(x: number): number {
  const z = x * x
  let num = ERF_A[4] * z
  let den = z
  for (let i = 0; i < 3; i++) {
    num = (num + ERF_A[i]) * z
    den = (den + ERF_B[i]) * z
  }
  return (x * (num + ERF_A[3])) / (den + ERF_B[3])
}

/** erfc(y) for y > 0.46875 (with the split exp(-y²) trick for accuracy). */
function erfcLarge(y: number): number {
  let result: number
  if (y <= 4) {
    let num = ERF_C[8] * y
    let den = y
    for (let i = 0; i < 7; i++) {
      num = (num + ERF_C[i]) * y
      den = (den + ERF_D[i]) * y
    }
    result = (num + ERF_C[7]) / (den + ERF_D[7])
  } else {
    const z = 1 / (y * y)
    let num = ERF_P[5] * z
    let den = z
    for (let i = 0; i < 4; i++) {
      num = (num + ERF_P[i]) * z
      den = (den + ERF_Q[i]) * z
    }
    result = (z * (num + ERF_P[4])) / (den + ERF_Q[4])
    result = (SQRPI - result) / y
  }
  const ysq = Math.trunc(y * 16) / 16
  const del = (y - ysq) * (y + ysq)
  return Math.exp(-ysq * ysq) * Math.exp(-del) * result
}

/** Complementary error function, double precision over the whole line. */
export function erfc(x: number): number {
  const y = Math.abs(x)
  if (y <= THRESH) return 1 - erfSmall(x)
  const tail = y > 26.6 ? 0 : erfcLarge(y)
  return x < 0 ? 2 - tail : tail
}

/** Standard normal CDF Φ(t), accurate relative to its size in both tails. */
export function ndtr(t: number): number {
  return 0.5 * erfc(-t / Math.SQRT2)
}

const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI)

/**
 * log Φ(x) with full relative accuracy everywhere.
 *
 * Below x ≈ -37.5, erfc underflows and the Mills-ratio series takes over:
 * Φ(x) = φ(x)/(-x) · (1 - 1/x² + 3/x⁴ - 15/x⁶ + …); at the handover point
 * 1/x² < 8e-4, so four terms are already beyond double precision.
 */
export function logNdtr(x: number): number {
  if (x > 6) {
    // Φ(x) = 1 - Q with Q tiny; log1p keeps all the accuracy Q has.
    return Math.log1p(-0.5 * erfc(x / Math.SQRT2))
  }
  if (x > -37) {
    return Math.log(0.5 * erfc(-x / Math.SQRT2))
  }
  const t = 1 / (x * x)
  const series = -t * (1 - 3 * t * (1 - 5 * t * (1 - 7 * t)))
  return -0.5 * x * x - Math.log(-x) - HALF_LOG_2PI + Math.log1p(series)
}

const ACK_A = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239]
const ACK_B = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
const ACK_C = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
const ACK_D = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]

const SQRT_2PI = Math.sqrt(2 * Math.PI)

/** Standard normal quantile Φ⁻¹(p) for p in (0, 1). */
export function ndtri(p: number): number {
  const pLow = 0.02425
  let x: number
  if (p >= pLow && p <= 1 - pLow) {
    const q = p - 0.5
    const r = q * q
    x =
      ((((((ACK_A[0] * r + ACK_A[1]) * r + ACK_A[2]) * r + ACK_A[3]) * r + ACK_A[4]) * r + ACK_A[5]) * q) /
      (((((ACK_B[0] * r + ACK_B[1]) * r + ACK_B[2]) * r + ACK_B[3]) * r + ACK_B[4]) * r + 1)
  } else {
    const lower = p < pLow
    const q = Math.sqrt(-2 * Math.log(lower ? p : 1 - p))
    x =
      (((((ACK_C[0] * q + ACK_C[1]) * q + ACK_C[2]) * q + ACK_C[3]) * q + ACK_C[4]) * q + ACK_C[5]) /
      ((((ACK_D[0] * q + ACK_D[1]) * q + ACK_D[2]) * q + ACK_D[3]) * q + 1)
    if (!lower) x = -x
  }
  // One Halley step against the accurate Φ, where its pieces are representable.
  if (p <= 1 - 1e-9 && x > -37.5) {
    const e = ndtr(x) - p
    const u = e * SQRT_2PI * Math.exp((x * x) / 2)
    x -= u / (1 + (x * u) / 2)
  }
  return x
}

/**
 * Φ⁻¹(exp(y)) for y ≤ 0 — scipy.special.ndtri_exp.
 *
 * Acklam (through exp) seeds the answer while exp(y) is representable; below
 * that the leading asymptotic inversion x ≈ -√(-2y) does. Newton on
 * log Φ(x) = y then polishes to machine precision: the update needs only
 * exp(logNdtr(x) - logPdf(x)) = Φ(x)/φ(x), which stays bounded.
 */
export function ndtriExp(y: number): number {
  if (y >= 0) return Infinity
  if (y === -Infinity) return -Infinity
  let x: number
  if (y > -690) {
    x = ndtri(Math.exp(y))
  } else {
    x = -Math.sqrt(-2 * y)
  }
  for (let iter = 0; iter < 3; iter++) {
    const logPhi = logNdtr(x)
    const logPdf = -0.5 * x * x - HALF_LOG_2PI
    const step = (logPhi - y) * Math.exp(logPhi - logPdf)
    x -= step
    if (Math.abs(step) < 1e-14 * Math.max(1, Math.abs(x))) break
  }
  return x
}
