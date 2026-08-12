/**
 * Asymmetric Numeral Systems (rANS), ported from the pure-Python reference in
 * simple_ans (github.com/flatironinstitute/simple_ans,
 * `simple_ans/pure_python/py_encode_decode.py`).
 *
 * The port is deliberately faithful — same 64-bit state, same 32-bit
 * renormalisation word, same `choose_symbol_counts`, same automatic choice of
 * precision — so its output is byte-for-byte what simple_ans produces, and can
 * be checked against it.
 *
 * The 64-bit state does not fit a JS number, so it is carried as a BigInt.
 */

const STATE_BITS = 64n
const WORD_BITS = 32n
const THRESHOLD = 1n << (STATE_BITS - WORD_BITS)
const MASK_WORD = (1n << WORD_BITS) - 1n

export interface EncodedSignal {
  state: bigint
  words: Uint32Array
  symbolCounts: Uint32Array
  symbolValues: Int16Array
  signalLength: number
  /** Bits used for the quantised symbol distribution; L = 2^precision. */
  precision: number
}

/**
 * Turn real-valued proportions into integer counts summing to L, each >= 1,
 * by largest remainder. Mirrors `choose_symbol_counts`.
 */
export function chooseSymbolCounts(proportions: Float64Array, L: number): Uint32Array {
  const k = proportions.length
  if (k > L) throw new Error('Number of proportions cannot exceed total items to distribute.')

  let total = 0
  for (const p of proportions) total += p

  const counts = new Uint32Array(k).fill(1)
  const remainder = L - k
  if (remainder > 0) {
    const frac = new Float64Array(k)
    let floorSum = 0
    for (let i = 0; i < k; i++) {
      const x = (remainder * proportions[i]) / total
      const f = Math.floor(x)
      counts[i] += f
      floorSum += f
      frac[i] = x - f
    }
    // Hand the leftover to the largest fractional parts.
    let leftover = remainder - floorSum
    if (leftover > 0) {
      const order = Array.from({ length: k }, (_, i) => i).sort((a, b) => frac[b] - frac[a])
      for (let i = 0; i < leftover; i++) counts[order[i]] += 1
    }
  }
  return counts
}

/** Sorted distinct values of the signal, with their counts. */
function histogram(signal: Int16Array): { values: Int16Array; counts: Float64Array } {
  const map = new Map<number, number>()
  for (const v of signal) map.set(v, (map.get(v) ?? 0) + 1)
  const values = Int16Array.from([...map.keys()].sort((a, b) => a - b))
  const counts = new Float64Array(values.length)
  for (let i = 0; i < values.length; i++) counts[i] = map.get(values[i])!
  return { values, counts }
}

function entropyBits(probs: Float64Array, against: Float64Array): number {
  let h = 0
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > 0) h -= probs[i] * Math.log2(against[i])
  }
  return h
}

/**
 * Smallest precision whose quantised distribution costs no more than 1/0.98 of
 * the true entropy. Mirrors the `precision is None` branch of `py_ans_encode`.
 */
function choosePrecision(probs: Float64Array, symbolCount: number): number {
  const target = entropyBits(probs, probs)
  for (let precision = 2; precision < 24; precision++) {
    const L = 2 ** precision
    if (L < symbolCount) continue
    const counts = chooseSymbolCounts(probs, L)
    const quantised = Float64Array.from(counts, c => c / L)
    if (entropyBits(probs, quantised) <= target / 0.98 || L >= 2 ** 20) return precision
  }
  return 23
}

export function ansEncode(signal: Int16Array, precisionOverride?: number): EncodedSignal {
  const { values, counts } = histogram(signal)
  const n = signal.length
  const probs = Float64Array.from(counts, c => c / n)

  const precision = precisionOverride ?? choosePrecision(probs, values.length)
  const L = 2 ** precision
  if (values.length > L) {
    throw new Error(`${values.length} distinct symbols exceeds index size ${L}`)
  }
  const symbolCounts = chooseSymbolCounts(probs, L)

  // Cumulative counts, and value -> symbol index.
  const cum = new Uint32Array(values.length)
  for (let i = 1; i < values.length; i++) cum[i] = cum[i - 1] + symbolCounts[i - 1]
  const index = new Map<number, number>()
  for (let i = 0; i < values.length; i++) index.set(values[i], i)

  const precisionN = BigInt(precision)
  const shift = STATE_BITS - precisionN
  let state = 0n
  const words: number[] = []

  for (let i = 0; i < n; i++) {
    const s = index.get(signal[i])!
    const F = BigInt(symbolCounts[s])
    // Renormalise: emit the low word so the state stays under 2^64.
    if (state >> shift >= F) {
      words.push(Number(state & MASK_WORD))
      state >>= WORD_BITS
    }
    state = ((state / F) << precisionN) | (BigInt(cum[s]) + (state % F))
  }

  return {
    state,
    words: Uint32Array.from(words),
    symbolCounts,
    symbolValues: values,
    signalLength: n,
    precision,
  }
}

export function ansDecode(e: EncodedSignal): Int16Array {
  const k = e.symbolCounts.length
  const cum = new Uint32Array(k)
  for (let i = 1; i < k; i++) cum[i] = cum[i - 1] + e.symbolCounts[i - 1]

  const L = 2 ** e.precision
  // quantile -> symbol index, so the decoder does a lookup rather than a scan.
  const slot = new Uint32Array(L)
  for (let s = 0; s < k; s++) {
    for (let j = 0; j < e.symbolCounts[s]; j++) slot[cum[s] + j] = s
  }

  const precisionN = BigInt(e.precision)
  const quantileMask = (1n << precisionN) - 1n
  const out = new Int16Array(e.signalLength)
  let state = e.state
  let stack = e.words.length - 1

  for (let i = 0; i < e.signalLength; i++) {
    const quantile = Number(state & quantileMask)
    const s = slot[quantile]
    let previous = (state >> precisionN) * BigInt(e.symbolCounts[s]) + BigInt(quantile - cum[s])
    if (previous < THRESHOLD && stack >= 0) {
      previous = (previous << WORD_BITS) | BigInt(e.words[stack--])
    }
    state = previous
    out[e.signalLength - i - 1] = e.symbolValues[s]
  }
  return out
}

/**
 * Bytes an encoded signal occupies: the state, the emitted words, and the
 * symbol table that the decoder needs. Matches `EncodedSignal.size()`.
 */
export function encodedSize(e: EncodedSignal): number {
  return 8 + e.words.byteLength + e.symbolCounts.byteLength + e.symbolValues.byteLength + 8
}
