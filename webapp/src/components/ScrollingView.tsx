import { useEffect, useRef, useState } from 'react'
import { LatentSource, LATENT_SEED } from '../model/latent'

/** Display samples generated per second while playing; px per sample fixed. */
const RATE = 220
const PX_PER_SAMPLE = 2

/** A nice round gridline step ≤ span/2. */
function niceStep(span: number): number {
  const raw = span / 2
  const mag = 10 ** Math.floor(Math.log10(raw))
  for (const m of [5, 2.5, 2, 1]) if (m * mag <= raw) return m * mag
  return mag
}

/**
 * A window of the quantized signal z, drawn as connected line segments.
 * The underlying randomness is a fixed latent sequence indexed by absolute
 * sample position — parameter changes re-render the same window of latent
 * data (no resampling), so the trace morphs smoothly. Stationary by default;
 * the play toggle advances the window through the latent sequence.
 */
export default function ScrollingView(props: {
  kernel: Float64Array
  sigma: number
  /** Predicted std of the quantized signal, for a stable y-scale. */
  sigmaY: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [playing, setPlaying] = useState(false)
  const playingRef = useRef(playing)
  playingRef.current = playing
  // Latent noise and window position survive parameter changes.
  const latentRef = useRef<LatentSource | null>(null)
  if (!latentRef.current) latentRef.current = new LatentSource(LATENT_SEED)
  const posRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const latent = latentRef.current!

    const scale = Math.max(4 * props.sigmaY, 3.5)
    const gridStep = niceStep(scale)

    let styles = getComputedStyle(canvas)
    const scheme = window.matchMedia('(prefers-color-scheme: dark)')
    const refreshStyles = () => {
      styles = getComputedStyle(canvas)
    }
    scheme.addEventListener('change', refreshStyles)

    let raf = 0
    let last = performance.now()
    let carry = 0

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw)
      const dt = Math.min(0.25, (now - last) / 1000)
      last = now
      if (playingRef.current) {
        carry += dt * RATE
        const n = Math.floor(carry)
        carry -= n
        posRef.current += n
      } else {
        carry = 0
      }

      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      if (w === 0 || h === 0) return
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr)
        canvas.height = Math.round(h * dpr)
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      const visible = Math.floor(w / PX_PER_SAMPLE)
      // The window ends at posRef and never reaches before index 0, so the
      // first thing shown is the start of the compression block.
      if (posRef.current < visible) posRef.current = visible
      const win = latent.window(posRef.current - visible, visible, props.kernel, props.sigma)

      const surface = styles.getPropertyValue('--surface')
      ctx.fillStyle = surface
      ctx.fillRect(0, 0, w, h)

      const yOf = (v: number) => h / 2 - (v / scale) * (h / 2 - 12)

      ctx.lineWidth = 1
      ctx.font = '11px system-ui, sans-serif'
      ctx.textAlign = 'left'
      for (let g = -2; g <= 2; g++) {
        const v = g * gridStep
        if (Math.abs(v) > scale) continue
        const y = Math.round(yOf(v)) + 0.5
        if (g !== 0) {
          ctx.strokeStyle = styles.getPropertyValue('--grid')
          ctx.beginPath()
          ctx.moveTo(0, y)
          ctx.lineTo(w, y)
          ctx.stroke()
        }
        // A surface-colored halo keeps the label readable over the trace.
        const label = `${v > 0 ? '+' : ''}${+v.toPrecision(3)}`
        ctx.fillStyle = styles.getPropertyValue('--muted')
        ctx.strokeStyle = surface
        ctx.lineWidth = 3
        ctx.strokeText(label, 6, y - 4)
        ctx.fillText(label, 6, y - 4)
        ctx.lineWidth = 1
      }
      const zeroY = Math.round(yOf(0)) + 0.5
      ctx.strokeStyle = styles.getPropertyValue('--baseline')
      ctx.beginPath()
      ctx.moveTo(0, zeroY)
      ctx.lineTo(w, zeroY)
      ctx.stroke()

      ctx.strokeStyle = styles.getPropertyValue('--series-1')
      ctx.lineWidth = 2
      ctx.lineJoin = 'round'
      ctx.beginPath()
      for (let i = 0; i < visible; i++) {
        const x = w - (visible - i) * PX_PER_SAMPLE + PX_PER_SAMPLE / 2
        const y = yOf(win[i])
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()
    }
    raf = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(raf)
      scheme.removeEventListener('change', refreshStyles)
    }
  }, [props.kernel, props.sigma, props.sigmaY])

  return (
    <div className="scroll-wrap">
      <canvas ref={canvasRef} className="scroll-canvas" />
      <button className="play-btn" onClick={() => setPlaying(p => !p)}>
        {playing ? '⏸ pause' : '▶ play'}
      </button>
    </div>
  )
}
