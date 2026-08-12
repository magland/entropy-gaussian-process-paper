/**
 * The particle-filter entropy-rate estimator off the main thread: one message
 * starts an endless refinement loop that posts replicates as they complete
 * (with per-step progress in between), exactly the streaming loop of
 * egp/estimate.py. Each replicate draws a fresh sequence and runs a fresh
 * filter over it, so replicates are i.i.d. and their spread is an honest
 * standard error. Stopping is worker termination — the loop never yields, but
 * outgoing messages still flow, and replicates already delivered survive on
 * the main thread, so a later run resumes at startRep with the same derived
 * seeds as an uninterrupted one.
 *
 * Two engines: the scalar filter of pf.ts, and the optional WebGPU filter of
 * pfGpu.ts, which runs a batch of replicates concurrently (one workgroup
 * each). Both consume identical y sequences — generation always uses the same
 * seeded CPU RNG — so the engines differ only in the filter's internal
 * randomness and are directly comparable.
 */
import { particleFilterLoglik, type FilterResult } from '../egp/pf'
import { GpuParticleFilter } from '../egp/pfGpu'
import { generateQuantized, replicateSeed } from '../egp/process'
import { Rng } from '../egp/rng'

export type Engine = 'cpu' | 'gpu'

export interface EntropyRequest {
  /** Minimum-phase taps of the scaled process (from the model worker). */
  taps: Float64Array
  delta: number
  /** Sequence length per replicate. */
  n: number
  nParticles: number
  seed: number
  /** Index of the first replicate to compute (count already done). */
  startRep: number
  engine: Engine
}

export interface RepResult {
  index: number
  entropyRateBits: number
  meanEss: number
  minEss: number
  lostLockFraction: number
  seconds: number
}

export type EntropyUpdate =
  | { type: 'progress'; rep: number; batch: number; done: number; total: number }
  | { type: 'rep'; result: RepResult }
  | { type: 'error'; message: string }

const post = self.postMessage as (message: EntropyUpdate) => void

function toRep(index: number, result: FilterResult, seconds: number): RepResult {
  return {
    index,
    entropyRateBits: result.entropyRateBits,
    meanEss: result.meanEss,
    minEss: result.minEss,
    lostLockFraction: result.lostLockFraction,
    seconds,
  }
}

function runCpu(req: EntropyRequest): void {
  const { taps, delta, n, nParticles, seed, startRep } = req
  for (let i = startRep; ; i++) {
    const started = performance.now()
    const rng = new Rng(replicateSeed(seed, i))
    const y = generateQuantized(taps, n, delta, rng)
    post({ type: 'progress', rep: i, batch: 1, done: 0, total: n })
    try {
      const result = particleFilterLoglik(y, taps, delta, nParticles, rng, undefined, (done, total) =>
        post({ type: 'progress', rep: i, batch: 1, done, total }),
      )
      post({ type: 'rep', result: toRep(i, result, (performance.now() - started) / 1000) })
    } catch (err) {
      // Total collapse (every particle at zero probability): report and stop
      // refining — more replicates at this N would all end the same way.
      post({ type: 'error', message: String(err) })
      return
    }
  }
}

async function runGpu(req: EntropyRequest): Promise<void> {
  const { taps, delta, n, nParticles, seed, startRep } = req
  let gpu: GpuParticleFilter
  try {
    gpu = await GpuParticleFilter.create()
  } catch (err) {
    post({ type: 'error', message: `WebGPU unavailable (${err}) — switch compute to CPU` })
    return
  }
  try {
    for (let batch = startRep; ; ) {
      const R = gpu.maxReps(nParticles, taps.length - 1, n)
      const repSeeds = new Uint32Array(R)
      const ys: Int32Array[] = []
      for (let r = 0; r < R; r++) {
        repSeeds[r] = replicateSeed(seed, batch + r)
        // Same generator and seed as the CPU engine: identical sequences.
        ys.push(generateQuantized(taps, n, delta, new Rng(repSeeds[r])))
      }
      post({ type: 'progress', rep: batch, batch: R, done: 0, total: n })
      const started = performance.now()
      const results = await gpu.run({
        taps,
        delta,
        ys,
        nParticles,
        repSeeds,
        onProgress: (done, total) => post({ type: 'progress', rep: batch, batch: R, done, total }),
      })
      const perRep = (performance.now() - started) / 1000 / R
      results.forEach((result, r) => post({ type: 'rep', result: toRep(batch + r, result, perRep) }))
      batch += R
    }
  } catch (err) {
    post({ type: 'error', message: `WebGPU run failed (${err}) — switch compute to CPU` })
  } finally {
    gpu.destroy()
  }
}

self.onmessage = (e: MessageEvent<EntropyRequest>) => {
  // The white process (L = 1) is exact and instant on the CPU regardless.
  if (e.data.engine === 'gpu' && e.data.taps.length > 1) {
    void runGpu(e.data)
  } else {
    runCpu(e.data)
  }
}
