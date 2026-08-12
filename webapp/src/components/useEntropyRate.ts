import { useEffect, useRef, useState } from 'react'
import { LOST_LOCK_TOLERANCE } from '../egp/pf'
import type { Engine, EntropyRequest, EntropyUpdate, RepResult } from '../worker/entropyWorker'

/** Fixed base seed: resumed runs continue the same per-replicate seed
 * sequence, so a stop/start pair reproduces an uninterrupted run exactly. */
const BASE_SEED = 20260806

export interface EntropyRate {
  /** One completed replicate per entry, in order — failures included. */
  reps: RepResult[]
  /** Mean over the replicates that kept lock; null while none have. */
  mean: number | null
  se: number | null
  /** Mean effective sample size over those replicates, fraction of N. */
  meanEss: number | null
  /** Replicates discarded for losing the state, and how many survived. */
  nFailed: number
  nUsable: number
  running: boolean
  /** "rep 3 · 38%" (or "reps 1–8 · 38%" for a GPU batch) while computing. */
  progress: string | null
  error: string | null
  start: () => void
  stop: () => void
}

/**
 * The in-browser particle-filter estimate: a worker refines it (one
 * independent replicate at a time, streaming per-step progress) until
 * stopped, and any change to the model or the filter settings invalidates
 * both the values and a run in flight.
 */
export function useEntropyRate(
  taps: Float64Array | null,
  n: number,
  nParticles: number,
  engine: Engine,
): EntropyRate {
  const [reps, setReps] = useState<RepResult[]>([])
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const repsRef = useRef<RepResult[]>([])

  useEffect(() => {
    workerRef.current?.terminate()
    workerRef.current = null
    repsRef.current = []
    setReps([])
    setRunning(false)
    setProgress(null)
    setError(null)
  }, [taps, n, nParticles, engine])

  useEffect(() => () => workerRef.current?.terminate(), [])

  const start = () => {
    if (workerRef.current || !taps) return
    const worker = new Worker(new URL('../worker/entropyWorker.ts', import.meta.url), {
      type: 'module',
    })
    worker.onmessage = (e: MessageEvent<EntropyUpdate>) => {
      const u = e.data
      if (u.type === 'rep') {
        repsRef.current = [...repsRef.current, u.result]
        setReps(repsRef.current)
      } else if (u.type === 'progress') {
        const pct = `${Math.round((100 * u.done) / u.total)}%`
        const label = u.batch > 1 ? `reps ${u.rep + 1}–${u.rep + u.batch}` : `rep ${u.rep + 1}`
        setProgress(`${label} · ${pct}`)
      } else {
        setError(u.message)
        setRunning(false)
        setProgress(null)
        worker.terminate()
        workerRef.current = null
      }
    }
    workerRef.current = worker
    const request: EntropyRequest = {
      taps,
      delta: 1,
      n,
      nParticles,
      seed: BASE_SEED,
      startRep: repsRef.current.length,
      engine,
    }
    worker.postMessage(request)
    setRunning(true)
    setError(null)
    setProgress(null)
  }

  const stop = () => {
    workerRef.current?.terminate()
    workerRef.current = null
    setRunning(false)
    setProgress(null)
  }

  // A replicate that lost the state contributes an arbitrary number rather
  // than a noisy one — one of them would move the mean by orders of
  // magnitude — so the average is over the survivors, and the discards are
  // counted and surfaced instead.
  const usable = reps.filter(r => r.lostLockFraction <= LOST_LOCK_TOLERANCE)
  const count = usable.length
  const values = usable.map(r => r.entropyRateBits)
  const mean = count > 0 ? values.reduce((a, b) => a + b, 0) / count : null
  let se: number | null = null
  if (mean !== null && count > 1) {
    const v = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (count - 1)
    se = Math.sqrt(v / count)
  }
  const meanEss = count > 0 ? usable.reduce((a, r) => a + r.meanEss, 0) / count : null

  return {
    reps,
    mean,
    se,
    meanEss,
    nFailed: reps.length - count,
    nUsable: count,
    running,
    progress,
    error,
    start,
    stop,
  }
}
