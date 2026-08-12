import { useEffect, useRef, useState } from 'react'
import type { ModelAnalysis, ModelRequest, ModelResponse } from '../worker/modelWorker'

export interface ModelState {
  /** Analysis of the *current* parameters, or the last completed one while a
   * newer request is in flight (so readouts don't flicker to placeholders). */
  analysis: ModelAnalysis | null
  computing: boolean
  error: string | null
}

/**
 * The minimum-phase factorization and spectral approximation for the current
 * (kernel, σ), computed in a worker: the factorization runs four FFTs on a
 * 2^18 grid, which is too slow for a slider's render path.
 */
export function useModelAnalysis(kernel: Float64Array, sigma: number): ModelState {
  const [state, setState] = useState<ModelState>({ analysis: null, computing: true, error: null })
  const workerRef = useRef<Worker | null>(null)
  const idRef = useRef(0)

  useEffect(() => {
    const worker = new Worker(new URL('../worker/modelWorker.ts', import.meta.url), {
      type: 'module',
    })
    worker.onmessage = (e: MessageEvent<ModelResponse>) => {
      if (e.data.id !== idRef.current) return
      setState({
        analysis: e.data.analysis ?? null,
        computing: false,
        error: e.data.error ?? null,
      })
    }
    workerRef.current = worker
    return () => {
      worker.terminate()
      workerRef.current = null
    }
  }, [])

  useEffect(() => {
    setState(s => ({ ...s, computing: true }))
    const id = ++idRef.current
    const timer = setTimeout(() => {
      const request: ModelRequest = { id, kernel, sigma }
      workerRef.current?.postMessage(request)
    }, 120)
    return () => clearTimeout(timer)
  }, [kernel, sigma])

  return state
}
