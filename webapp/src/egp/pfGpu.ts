/**
 * WebGPU implementation of the fully adapted particle filter — the optional
 * fast path beside the scalar pf.ts, same algorithm, same diagnostics.
 *
 * Design: one workgroup of 256 threads runs one replicate, looping over the
 * step sequence inside the shader with workgroup barriers between the phases
 * of a step (weights → max/scan reductions → systematic resample by binary
 * search → truncated-normal propagation). Several replicates therefore run
 * concurrently as independent workgroups of one dispatch, which is where the
 * speedup comes from — the per-step phases are sequential, so a single
 * replicate can only use one workgroup's worth of parallelism.
 *
 * WGSL is f32-only, so the log-space normal functions are f32 rebuilds, not
 * ports: Φ through the Abramowitz–Stegun erf (1.5e-7 absolute) above
 * x = −1.5 and the Laplace continued fraction for the Mills ratio below it;
 * Φ⁻¹(eʸ) seeded by Acklam's branches — the tails computed from log p and
 * −log p directly, where exp would cancel — and polished by Newton on log Φ,
 * skipped above x = 5 where f32 log Φ saturates to 0 and would corrupt the
 * seed. Weight errors land near 1e-4 relative and draw errors near 1e-2 at
 * 5σ, both far below the estimator's statistical resolution; per-step
 * log-likelihoods are read back as f32 and accumulated in f64 on the CPU,
 * where the lost-lock diagnostic is also evaluated, exactly as in pf.ts.
 *
 * Ancestor gathers shift the innovation window as they copy, so the tap
 * vector never rolls; ring buffers ping-pong on step parity. The RNG is a
 * counter-based PCG hash of (seed, stream, counter) — deterministic for a
 * given seed regardless of GPU scheduling.
 */
import { collapseThreshold, DEFAULT_COLLAPSE_MARGIN, type FilterResult } from './pf'
// The shader is shared verbatim with the Python package (egp/pf_gpu.py):
// one WGSL source, two hosts.
import SHADER from '../../../egp/src/egp/pf.wgsl?raw'

/** Concurrent replicates per dispatch; more gives diminishing returns for a
 * live readout and multiplies buffer sizes. */
const MAX_REPS = 8

export interface GpuRunOptions {
  taps: Float64Array
  delta: number
  /** One quantized sequence per replicate, all the same length. */
  ys: Int32Array[]
  nParticles: number
  /** One RNG seed per replicate for the filter's internal draws. */
  repSeeds: Uint32Array
  onProgress?: (done: number, total: number) => void
}

export class GpuParticleFilter {
  private constructor(
    private device: GPUDevice,
    private pipeline: GPUComputePipeline,
  ) {}

  static supported(): boolean {
    return typeof navigator !== 'undefined' && !!navigator.gpu
  }

  static async create(): Promise<GpuParticleFilter> {
    if (!GpuParticleFilter.supported()) throw new Error('WebGPU is not available here')
    const adapter = await navigator.gpu!.requestAdapter()
    if (!adapter) throw new Error('no WebGPU adapter available')
    const device = await adapter.requestDevice()
    const module = device.createShaderModule({ code: SHADER })
    const pipeline = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
    })
    return new GpuParticleFilter(device, pipeline)
  }

  destroy(): void {
    this.device.destroy()
  }

  /** Replicates one dispatch can run concurrently within buffer limits. */
  maxReps(nParticles: number, ringLen: number, nSteps: number): number {
    const limit = this.device.limits.maxStorageBufferBindingSize
    const perRep = Math.max(nParticles * ringLen * 4, nParticles * 16, nSteps * 8)
    return Math.max(1, Math.min(MAX_REPS, Math.floor(limit / Math.max(perRep, 1))))
  }

  /** Run one batch of replicates; resolves to one FilterResult per replicate. */
  async run(opts: GpuRunOptions): Promise<FilterResult[]> {
    const { taps, delta, ys, nParticles, repSeeds, onProgress } = opts
    const R = ys.length
    const n = ys[0].length
    const N = nParticles | 0
    const L = taps.length
    const ringLen = L - 1
    const h0 = taps[0]
    if (ringLen < 1) throw new Error('GPU path needs L >= 2 (white noise is exact on the CPU)')
    if (R < 1 || repSeeds.length !== R) throw new Error('one seed per replicate required')
    const { device, pipeline } = this

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    const ringBytes = R * N * ringLen * 4
    const buffers = {
      uniform: device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      ringA: device.createBuffer({ size: ringBytes, usage: storage }),
      ringB: device.createBuffer({ size: ringBytes, usage: storage }),
      ys: device.createBuffer({ size: R * n * 4, usage: storage }),
      stepOut: device.createBuffer({ size: R * n * 8, usage: storage | GPUBufferUsage.COPY_SRC }),
      scratch: device.createBuffer({ size: R * N * 16, usage: storage }),
      cum: device.createBuffer({ size: R * N * 4, usage: storage }),
      taps: device.createBuffer({ size: L * 4, usage: storage }),
      repSeeds: device.createBuffer({ size: R * 4, usage: storage }),
      readback: device.createBuffer({ size: R * n * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
    }
    try {
      const yAll = new Int32Array(R * n)
      ys.forEach((y, r) => yAll.set(y, r * n))
      device.queue.writeBuffer(buffers.ys, 0, yAll)
      device.queue.writeBuffer(buffers.taps, 0, new Float32Array(taps))
      device.queue.writeBuffer(buffers.repSeeds, 0, new Uint32Array(repSeeds))

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.uniform } },
          { binding: 1, resource: { buffer: buffers.ringA } },
          { binding: 2, resource: { buffer: buffers.ringB } },
          { binding: 3, resource: { buffer: buffers.ys } },
          { binding: 4, resource: { buffer: buffers.stepOut } },
          { binding: 5, resource: { buffer: buffers.scratch } },
          { binding: 6, resource: { buffer: buffers.cum } },
          { binding: 7, resource: { buffer: buffers.taps } },
          { binding: 8, resource: { buffer: buffers.repSeeds } },
        ],
      })

      // Steps per submit, sized to keep one dispatch well under any device
      // timeout while amortizing the submit/await round trip.
      const chunk = Math.max(8, Math.min(512, Math.round(3e7 / (N * (ringLen + 30)))))
      const uniform = new ArrayBuffer(48)
      const u32 = new Uint32Array(uniform)
      const f32 = new Float32Array(uniform)
      u32[0] = N
      u32[1] = ringLen
      u32[2] = n
      f32[8] = h0
      f32[9] = delta / h0
      f32[10] = delta
      f32[11] = Math.log(N)

      for (let startStep = 0; startStep < n; startStep += chunk) {
        u32[3] = startStep
        u32[4] = Math.min(chunk, n - startStep)
        device.queue.writeBuffer(buffers.uniform, 0, uniform)
        const encoder = device.createCommandEncoder()
        const pass = encoder.beginComputePass()
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroup)
        pass.dispatchWorkgroups(R)
        pass.end()
        device.queue.submit([encoder.finish()])
        await device.queue.onSubmittedWorkDone()
        onProgress?.(Math.min(startStep + chunk, n), n)
      }

      const encoder = device.createCommandEncoder()
      encoder.copyBufferToBuffer(buffers.stepOut, 0, buffers.readback, 0, R * n * 8)
      device.queue.submit([encoder.finish()])
      await buffers.readback.mapAsync(GPUMapMode.READ)
      const steps = new Float32Array(buffers.readback.getMappedRange().slice(0))
      buffers.readback.unmap()

      // Per-step values accumulate in f64 on the CPU, where the lost-lock
      // diagnostic is evaluated exactly as in the scalar filter.
      const limit = collapseThreshold(delta, h0, DEFAULT_COLLAPSE_MARGIN)
      return Array.from({ length: R }, (_, r) => {
        let loglik = 0
        let essSum = 0
        let essMin = 1
        let nLost = 0
        let firstLost = -1
        for (let t = 0; t < n; t++) {
          const step = steps[(r * n + t) * 2]
          const ess = steps[(r * n + t) * 2 + 1]
          loglik += step
          essSum += ess
          essMin = Math.min(essMin, ess)
          if (step < limit) {
            nLost++
            if (firstLost < 0) firstLost = t
          }
        }
        const result: FilterResult = {
          loglikNats: loglik,
          nSteps: n,
          entropyRateBits: -loglik / (n * Math.LN2),
          meanEss: essSum / n,
          minEss: essMin,
          nLostLock: nLost,
          firstLostStep: firstLost,
          lostLockFraction: nLost / n,
          collapseThreshold: limit,
        }
        return result
      })
    } finally {
      for (const b of Object.values(buffers)) b.destroy()
    }
  }
}
