import { useMemo, useRef, useState } from 'react'
import { magnitudeResponse } from '../model/filters'
import type { ModelAnalysis } from '../worker/modelWorker'
import { useWidth } from './useWidth'

const MARGIN = { left: 44, right: 12, top: 10, bottom: 24 }
const HEIGHT = 120
const DB_FLOOR = -80
const DB_TICKS = [0, -20, -40, -60, -80]
const RESPONSE_POINTS = 512

interface Tip {
  x: number
  y: number
  value: string
  label: string
}

function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null
  return (
    <div className="viz-tooltip" style={{ left: tip.x + 12, top: tip.y - 10 }}>
      <span className="tip-value">{tip.value}</span> <span className="tip-label">{tip.label}</span>
    </div>
  )
}

/** ~n round tick values covering [lo, hi]. */
function ticks(lo: number, hi: number, n: number): number[] {
  const span = hi - lo
  if (!(span > 0)) return [lo]
  const raw = span / n
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v)
  return out
}

function formatHz(hz: number): string {
  return hz >= 1000 ? `${+(hz / 1000).toFixed(1)}k` : `${Math.round(hz)}`
}

/** A tap sequence against time, shared by the design and min-phase panels. */
function TapsPanel({
  title,
  taps,
  sampleRateHz,
  centered,
  color,
}: {
  title: string
  taps: Float64Array
  sampleRateHz: number
  /** Zero-phase designs are drawn centered on t = 0; causal ones start there. */
  centered: boolean
  color: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const width = useWidth(ref)
  const [tip, setTip] = useState<Tip | null>(null)

  const L = taps.length
  const mid = centered ? (L - 1) / 2 : 0
  const msAt = (i: number) => ((i - mid) / sampleRateHz) * 1000
  // A one-tap kernel still needs a nonzero span to draw on.
  const tSpan = L > 1 ? msAt(L - 1) - msAt(0) : 1000 / sampleRateHz
  const t0 = msAt(0) - (L === 1 ? tSpan / 2 : 0)
  const plotW = width - MARGIN.left - MARGIN.right
  const plotH = HEIGHT - MARGIN.top - MARGIN.bottom
  let yMin = 0
  let yMax = 0
  for (const v of taps) {
    yMin = Math.min(yMin, v)
    yMax = Math.max(yMax, v)
  }
  const pad = (yMax - yMin || 1) * 0.1
  yMin -= pad
  yMax += pad
  const xOf = (i: number) => MARGIN.left + ((msAt(i) - t0) / tSpan) * plotW
  const yOf = (v: number) => MARGIN.top + ((yMax - v) / (yMax - yMin)) * plotH

  const path = Array.from(taps, (v, i) => `${i ? 'L' : 'M'}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join('')

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - box.left
    const i = L === 1 ? 0 : Math.max(0, Math.min(L - 1, Math.round(((px - MARGIN.left) / plotW) * (L - 1))))
    setTip({
      x: px,
      y: e.clientY - box.top,
      value: taps[i].toPrecision(3),
      label: `tap ${i} · ${msAt(i).toFixed(2)} ms`,
    })
  }

  return (
    <div className="panel" ref={ref}>
      <h3>{title}</h3>
      <svg width={width} height={HEIGHT} onPointerMove={onMove} onPointerLeave={() => setTip(null)}>
        <line
          x1={MARGIN.left}
          x2={width - MARGIN.right}
          y1={yOf(0)}
          y2={yOf(0)}
          stroke="var(--baseline)"
          strokeWidth={1}
        />
        {ticks(t0, t0 + tSpan, 5)
          .filter(t => MARGIN.left + ((t - t0) / tSpan) * plotW < width - MARGIN.right - 34)
          .map(t => (
            <text key={t} x={MARGIN.left + ((t - t0) / tSpan) * plotW} y={HEIGHT - 8} textAnchor="middle" className="axis-tick">
              {+t.toFixed(2)}
            </text>
          ))}
        <text x={width - MARGIN.right} y={HEIGHT - 8} textAnchor="end" className="axis-tick">
          ms
        </text>
        {ticks(yMin, yMax, 3).map(v => (
          <text key={v} x={MARGIN.left - 6} y={yOf(v) + 4} textAnchor="end" className="axis-tick">
            {+v.toPrecision(2)}
          </text>
        ))}
        <path d={path} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {L <= 31 &&
          Array.from(taps, (v, i) => (
            <circle key={i} cx={xOf(i)} cy={yOf(v)} r={4} fill={color} stroke="var(--surface)" strokeWidth={2} />
          ))}
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

/** |H(f)| in dB up to Nyquist. */
function ResponsePanel({ kernel, sampleRateHz }: { kernel: Float64Array; sampleRateHz: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const width = useWidth(ref)
  const [tip, setTip] = useState<Tip | null>(null)

  const db = useMemo(() => {
    const mag = magnitudeResponse(kernel, RESPONSE_POINTS)
    return Array.from(mag, m => Math.max(DB_FLOOR, 20 * Math.log10(Math.max(m, 1e-12))))
  }, [kernel])

  const nyquist = sampleRateHz / 2
  const plotW = width - MARGIN.left - MARGIN.right
  const plotH = HEIGHT - MARGIN.top - MARGIN.bottom
  const dbMax = Math.max(5, Math.ceil(Math.max(...db) / 5) * 5)
  const yOf = (v: number) => MARGIN.top + ((dbMax - v) / (dbMax - DB_FLOOR)) * plotH
  const xOf = (i: number) => MARGIN.left + (i / (RESPONSE_POINTS - 1)) * plotW

  const path = db.map((v, i) => `${i ? 'L' : 'M'}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join('')

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - box.left
    const i = Math.max(0, Math.min(RESPONSE_POINTS - 1, Math.round(((px - MARGIN.left) / plotW) * (RESPONSE_POINTS - 1))))
    setTip({
      x: px,
      y: e.clientY - box.top,
      value: `${db[i].toFixed(1)} dB`,
      label: `at ${formatHz((i / (RESPONSE_POINTS - 1)) * nyquist)} Hz`,
    })
  }

  return (
    <div className="panel" ref={ref}>
      <h3>Frequency response |H(f)|</h3>
      <svg width={width} height={HEIGHT} onPointerMove={onMove} onPointerLeave={() => setTip(null)}>
        {DB_TICKS.map(v => (
          <g key={v}>
            <line x1={MARGIN.left} x2={width - MARGIN.right} y1={yOf(v)} y2={yOf(v)} stroke="var(--grid)" strokeWidth={1} />
            <text x={MARGIN.left - 6} y={yOf(v) + 4} textAnchor="end" className="axis-tick">
              {v}
            </text>
          </g>
        ))}
        <text x={MARGIN.left - 6} y={MARGIN.top - 1} textAnchor="end" className="axis-tick">
          dB
        </text>
        {ticks(0, nyquist, 5)
          // Leave the right corner to the unit label.
          .filter(f => MARGIN.left + (f / nyquist) * plotW < width - MARGIN.right - 34)
          .map(f => (
            <text key={f} x={MARGIN.left + (f / nyquist) * plotW} y={HEIGHT - 8} textAnchor="middle" className="axis-tick">
              {formatHz(f)}
            </text>
          ))}
        <text x={width - MARGIN.right} y={HEIGHT - 8} textAnchor="end" className="axis-tick">
          Hz
        </text>
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      </svg>
      <Tooltip tip={tip} />
    </div>
  )
}

export default function FilterViz(props: {
  kernel: Float64Array
  sampleRateHz: number
  sigma: number
  analysis: ModelAnalysis | null
}) {
  const { analysis, sigma } = props
  // The min-phase factor is computed on σ·kernel; divide it back out so the
  // two kernel panels share the design kernel's units.
  const minPhase = useMemo(() => {
    if (!analysis) return null
    const out = new Float64Array(analysis.minPhaseTaps.length)
    for (let i = 0; i < out.length; i++) out[i] = analysis.minPhaseTaps[i] / sigma
    return out
  }, [analysis, sigma])

  return (
    <div>
      <div className="filter-panels three">
        <TapsPanel
          title="Design kernel h (zero-phase)"
          taps={props.kernel}
          sampleRateHz={props.sampleRateHz}
          centered
          color="var(--series-1)"
        />
        {minPhase && (
          <TapsPanel
            title="Minimum-phase factor h̃ (same |H|)"
            taps={minPhase}
            sampleRateHz={props.sampleRateHz}
            centered={false}
            color="var(--series-4)"
          />
        )}
        <ResponsePanel kernel={props.kernel} sampleRateHz={props.sampleRateHz} />
      </div>
      {analysis && (
        <p className="card-note">
          The particle filter runs on the minimum-phase factor h̃ — the causal square root of the
          spectrum with all its energy up front, whose leading tap is the one-step prediction
          error (Szegő): here h₀ = σ·h̃₀ = {analysis.h0.toPrecision(4)} quantization steps, so
          Δ/h₀ = {(1 / analysis.h0).toPrecision(3)}. A zero-phase kernel defines the same process
          law but has a near-zero leading tap, which would make the filter degenerate.
          Factorization check: autocovariance mismatch {analysis.autocovError.toExponential(1)}{' '}
          of γ₀, spectrum floor touched {(100 * analysis.flooredFraction).toFixed(2)}% of the
          grid.
        </p>
      )}
    </div>
  )
}
