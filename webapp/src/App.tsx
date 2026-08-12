import { useEffect, useMemo, useRef, useState } from 'react'
import CopyableCommand from './components/CopyableCommand'
import Controls from './components/Controls'
import FilterViz from './components/FilterViz'
import ScrollingView from './components/ScrollingView'
import CompressionChart from './components/CompressionChart'
import MethodNote from './components/MethodNote'
import { useEntropyRate } from './components/useEntropyRate'
import { useModelAnalysis } from './components/useModelAnalysis'
import { DEFAULT_SPEC, clampSpec, designKernel, kernelNorm } from './model/filters'
import { LATENT_SEED } from './model/latent'
import { DEFAULT_LPC_ORDER, LPC_ORDERS } from './compress/codecs'
import type { CodecResult } from './compress/codecs'
import type { CompressRequest, CompressResponse } from './worker/compressWorker'
import type { Engine } from './worker/entropyWorker'
import { GpuParticleFilter } from './egp/pfGpu'
import { LOST_LOCK_TOLERANCE } from './egp/pf'

const BLOCK_SIZES = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
const DEFAULT_BLOCK_SIZE = 100000

/**
 * Particle counts N for the filter. The paper's point stands here too: N
 * cannot be fixed in advance — it must grow with L and σ/h₀ — so the default
 * is generous, headroom above it exists for the hard models, and the low end
 * is kept reachable, since watching the filter lose lock at N = 300 is the
 * most direct way to see why N matters.
 */
const PARTICLE_COUNTS = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
const DEFAULT_PARTICLES = 30000
/**
 * Sequence length n per replicate. Cost is O(nNL), so n trades directly
 * against N: the default spends the budget on particles, which fight the
 * collapse that makes an estimate meaningless, rather than on length, which
 * only shrinks an O(L/n) bias and the per-replicate noise that averaging over
 * replicates already handles.
 */
const SEQ_LENGTHS = [100, 300, 1000, 3000, 10000, 30000, 100000]
const DEFAULT_SEQ_LENGTH = 1000

/** Whether the optional WebGPU engine can even be offered here. The worker
 * re-checks for itself; this only controls the selector's presence. */
const GPU_AVAILABLE = GpuParticleFilter.supported()

interface CompressionState {
  results: CodecResult[]
  computing: boolean
  error: string | null
}

/** The nine codec sizes, measured in a worker on a debounced parameter set. */
function useCompression(
  kernel: Float64Array,
  sigma: number,
  lpcOrder: number,
  blockSize: number,
): CompressionState {
  const [state, setState] = useState<CompressionState>({
    results: [],
    computing: true,
    error: null,
  })
  const workerRef = useRef<Worker | null>(null)
  const idRef = useRef(0)

  useEffect(() => {
    const worker = new Worker(new URL('./worker/compressWorker.ts', import.meta.url), {
      type: 'module',
    })
    worker.onmessage = (e: MessageEvent<CompressResponse>) => {
      if (e.data.id !== idRef.current) return
      setState({
        results: e.data.error ? [] : e.data.results,
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
      const request: CompressRequest = {
        id,
        kernel,
        sigma,
        blockSize,
        lpcOrder,
        // The same seed the signal view draws from, so the block really is
        // the data on screen.
        seed: LATENT_SEED,
      }
      workerRef.current?.postMessage(request)
    }, 250)
    return () => clearTimeout(timer)
  }, [kernel, sigma, lpcOrder, blockSize])

  return state
}

/**
 * The terminal command that estimates R̄ at the current settings with the
 * reference implementation — the egp Python package in ../egp. The app's
 * pipeline rounds σ·(h∗x) to integers, which is egp's process at Δ = 1 with
 * the marginal std set to σ_Y = σ‖h‖.
 */
function egpCommand(
  spec: ReturnType<typeof clampSpec>,
  rate: number,
  sigmaY: number,
  n: number,
  nParticles: number,
): string {
  const parts = ['egp estimate']
  switch (spec.family) {
    case 'none':
      parts.push('--preset white')
      break
    case 'movingAverage':
      parts.push('--preset ma', `--taps ${spec.width}`)
      break
    case 'lowpass':
      parts.push('--preset lowpass', `--cutoff ${spec.highHz}`, `--fs ${rate}`, `--taps ${spec.taps}`)
      break
    case 'bandpass':
      parts.push(
        '--preset bandpass',
        `--band ${spec.lowHz} ${spec.highHz}`,
        `--fs ${rate}`,
        `--taps ${spec.taps}`,
      )
      break
    case 'firstDifference':
      parts.push('--preset diff')
      break
  }
  parts.push(`--sigma ${sigmaY.toPrecision(6)}`, '--delta 1', `-n ${n}`, `-N ${nParticles}`, '-r 3')
  return parts.join(' ')
}

export default function App() {
  const [sigma, setSigma] = useState(5)
  const [sampleRateHz, setSampleRateHz] = useState(30000)
  const [spec, setSpec] = useState(DEFAULT_SPEC)
  const [lpcOrder, setLpcOrder] = useState(DEFAULT_LPC_ORDER)
  const [blockSize, setBlockSize] = useState(DEFAULT_BLOCK_SIZE)
  const [nParticles, setNParticles] = useState(DEFAULT_PARTICLES)
  const [seqLength, setSeqLength] = useState(DEFAULT_SEQ_LENGTH)
  const [engine, setEngine] = useState<Engine>(GPU_AVAILABLE ? 'gpu' : 'cpu')

  const kernel = useMemo(() => designKernel(spec, sampleRateHz), [spec, sampleRateHz])
  const sigmaY = useMemo(() => sigma * kernelNorm(kernel), [kernel, sigma])
  const compression = useCompression(kernel, sigma, lpcOrder, blockSize)
  const model = useModelAnalysis(kernel, sigma)
  const analysis = model.analysis
  const entropy = useEntropyRate(analysis?.minPhaseTaps ?? null, seqLength, nParticles, engine)

  // Δ/h₀ governs the quantization regime; σ/h₀ (with L) governs how many
  // particles the filter needs — when h₀ ≪ σ the observations nearly
  // determine the state and an off-by-a-cell cloud cannot recover.
  const deltaOverH0 = analysis ? 1 / analysis.h0 : null
  const sigmaOverH0 = analysis ? analysis.sigmaY / analysis.h0 : null
  const regime =
    deltaOverH0 === null || sigmaOverH0 === null
      ? ''
      : sigmaOverH0 > 20
        ? 'hard for the filter — needs many particles'
        : deltaOverH0 > 3
          ? 'coarse quantization'
          : deltaOverH0 < 1 / 3
            ? 'fine quantization'
            : 'intermediate quantization'

  // Collapse verdicts, per replicate, from the per-step diagnostic that works
  // where ESS does not. Replicates that lost the state are already excluded
  // from the mean; what is left to decide is what to say about them.
  const allFailed = entropy.reps.length > 0 && entropy.nUsable === 0
  const someFailed = entropy.nFailed > 0 && entropy.nUsable > 0

  const repList = entropy.reps.map(r =>
    r.lostLockFraction > LOST_LOCK_TOLERANCE ? '✗' : r.entropyRateBits.toFixed(3),
  )
  const repSummary =
    repList.length > 0
      ? `reps ${repList.length > 6 ? '…' : ''}${repList.slice(-6).join(', ')}`
      : null

  return (
    <div className="app">
      <header className="app-header">
        <h1>Entropy rate of a quantized Gaussian process</h1>
        <p>
          Gaussian noise → FIR filter → round to integers. What is the entropy rate of the
          integer stream — the bits per sample no lossless code can beat — and how close do
          practical codecs get? The rate is estimated in the browser by a fully adapted particle
          filter on the minimum-phase form of the process, checked against an analytic spectral
          approximation.
        </p>
      </header>

      {/* The controls stay pinned so the parameters and the numbers they move
          are always on screen together, whatever is scrolled to. */}
      <section className="card control-bar">
        <Controls
          sigma={sigma}
          setSigma={setSigma}
          sampleRateHz={sampleRateHz}
          setSampleRateHz={rate => {
            // Band edges are absolute, so a new rate can push them past
            // Nyquist; re-snap the spec so sliders and kernel stay in step.
            setSampleRateHz(rate)
            setSpec(s => clampSpec(s, rate))
          }}
          spec={spec}
          setSpec={setSpec}
        />
      </section>

      <section className="card">
        <h2>Entropy rate and compression</h2>
        <div className="stat-row">
          <div className="stat">
            <span className="label">
              <span className="line-swatch" style={{ borderTop: '2px dotted var(--theory)' }} />
              R̄ — spectral approximation
            </span>
            <span className="value">
              {analysis ? analysis.approxBits.toFixed(2) : '—'} <small>bits/sample</small>
            </span>
            <span className="stat-sub">
              {analysis && analysis.approxBits > 0
                ? `best possible ratio ${(16 / analysis.approxBits).toFixed(2)}×`
                : '—'}
            </span>
          </div>
          <div className="stat">
            <span className="label">
              <span className="line-swatch" style={{ borderTop: '2px dashed var(--ink-2)' }} />
              R̄ — particle filter (SMB)
            </span>
            <span className="value">
              {entropy.mean !== null ? entropy.mean.toFixed(2) : '—'}
              {entropy.se !== null && <small> ± {entropy.se.toFixed(2)}</small>}{' '}
              <small>bits/sample</small>
            </span>
            <span className="stat-sub">
              {entropy.mean !== null && entropy.mean > 0
                ? `best possible ratio ${(16 / entropy.mean).toFixed(2)}×`
                : allFailed
                  ? 'no replicate kept lock'
                  : 'run the estimate to check the approximation'}
            </span>
            {/* The estimate lives with its readout: start, watch replicates
                stream in, stop; a model change resets it. */}
            <span className="stat-action">
              <button onClick={entropy.running ? entropy.stop : entropy.start} disabled={!analysis}>
                {entropy.running ? 'stop' : entropy.reps.length > 0 ? 'refine' : 'estimate'}
              </button>
              <span className="estimate-status">
                {entropy.running
                  ? `${entropy.progress ?? 'starting…'}${
                      entropy.meanEss !== null ? ` · ESS ${(100 * entropy.meanEss).toFixed(0)}%` : ''
                    }`
                  : entropy.reps.length > 0
                    ? `${entropy.nUsable} replicate${entropy.nUsable === 1 ? '' : 's'}${
                        entropy.nFailed > 0 ? ` (${entropy.nFailed} discarded)` : ''
                      } · ESS ${(100 * (entropy.meanEss ?? 0)).toFixed(0)}% · N = ${nParticles.toLocaleString()}`
                    : `N = ${nParticles.toLocaleString()} · n = ${seqLength.toLocaleString()}`}
              </span>
            </span>
            {repSummary && <span className="stat-sub muted-sub">{repSummary}</span>}
            {entropy.error && <span className="stat-warning">✗ {entropy.error}</span>}
            {!entropy.error && allFailed && (
              <span className="stat-warning">
                ⚠ every replicate lost lock — no estimate. Increase particles.
              </span>
            )}
            {!entropy.error && someFailed && (
              <span className="stat-warning">
                ⚠ {entropy.nFailed} of {entropy.reps.length} replicates lost lock and were
                discarded; the value averages the {entropy.nUsable} that survived. Provisional
                until it runs clean — increase particles.
              </span>
            )}
          </div>
          <div className="stat">
            <span className="label">prediction error h₀ (Szegő)</span>
            <span className="value">
              {analysis ? analysis.h0.toPrecision(3) : '—'} <small>quantization steps</small>
            </span>
            <span className="stat-sub">
              {deltaOverH0 !== null && sigmaOverH0 !== null
                ? `Δ/h₀ = ${deltaOverH0.toPrecision(3)} · σ/h₀ = ${sigmaOverH0.toPrecision(3)} · ${regime}`
                : '—'}
            </span>
          </div>
        </div>
        {/* Settings of the measurement, not of the model — so they live with
            the numbers they change rather than in the model bar. */}
        <div className="measure-row">
          <label>
            particles N
            <select value={nParticles} onChange={e => setNParticles(Number(e.target.value))}>
              {PARTICLE_COUNTS.map(v => (
                <option key={v} value={v}>
                  {v.toLocaleString()}
                </option>
              ))}
            </select>
          </label>
          <label>
            filter length n
            <select value={seqLength} onChange={e => setSeqLength(Number(e.target.value))}>
              {SEQ_LENGTHS.map(v => (
                <option key={v} value={v}>
                  {v.toLocaleString()} samples
                </option>
              ))}
            </select>
          </label>
          {GPU_AVAILABLE && (
            <label>
              compute
              <select value={engine} onChange={e => setEngine(e.target.value as Engine)}>
                <option value="gpu">WebGPU</option>
                <option value="cpu">CPU</option>
              </select>
            </label>
          )}
          <label>
            LPC order
            <select value={lpcOrder} onChange={e => setLpcOrder(Number(e.target.value))}>
              {LPC_ORDERS.map(o => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
          <label>
            block size
            <select value={blockSize} onChange={e => setBlockSize(Number(e.target.value))}>
              {BLOCK_SIZES.map(n => (
                <option key={n} value={n}>
                  {n.toLocaleString()} samples
                </option>
              ))}
            </select>
          </label>
        </div>
        {compression.error ? (
          <p className="card-note">Compression failed: {compression.error}</p>
        ) : (
          <CompressionChart
            results={compression.results}
            // entropy.mean already excludes the replicates that lost lock, so
            // it is either a real estimate or null (nothing survived).
            rateBits={entropy.mean}
            rateSe={entropy.se}
            approxBits={analysis?.approxBits ?? 0}
            computing={compression.computing || model.computing}
          />
        )}
        <p className="card-note">
          Measured on a {blockSize.toLocaleString()}-sample block of the same latent data the
          signal view shows; sizes include everything a decoder needs (ANS symbol table, LPC
          coefficients) and every reported size round-trips through the decoder. Baseline is raw
          int16 (16 bits/sample). The two reference lines mark the entropy rate R̄ of the
          process — the one limit no lossless method whatsoever can beat: dotted for the
          analytic spectral approximation, dashed for the particle-filter estimate, shaded by
          its standard error (see the method section at the bottom). Raw-symbol codecs cluster
          near the memoryless (order-0) entropy of the stream; prediction (delta, LPC) is what
          carries a codec toward R̄ — the gap that remains is the cost of coding the residual
          memorylessly.
        </p>
        <CopyableCommand
          label="cross-check R̄ with the Python reference (pip install -e ../egp):"
          command={egpCommand(clampSpec(spec, sampleRateHz), sampleRateHz, sigmaY, seqLength, nParticles)}
        />
      </section>

      <section className="card">
        <h2>Quantized signal z</h2>
        <ScrollingView kernel={kernel} sigma={sigma} sigmaY={sigmaY} />
        <p className="card-note">
          A window of samples from the model, drawn from a fixed latent noise sequence — changing
          σ or the filter transforms the same underlying data, so the trace morphs rather than
          resampling. Press play to advance through the sequence.
        </p>
      </section>

      <section className="card">
        <h2>Filter and its minimum-phase factor</h2>
        <FilterViz kernel={kernel} sampleRateHz={sampleRateHz} sigma={sigma} analysis={analysis} />
      </section>

      <section className="card">
        <h2>The entropy rate</h2>
        <MethodNote />
      </section>
    </div>
  )
}
