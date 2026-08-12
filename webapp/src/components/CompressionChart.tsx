import { useRef, useState } from 'react'
import type { CodecResult } from '../compress/codecs'
import { useWidth } from './useWidth'

const GROUPS = ['no prefilter', 'delta', 'LPC']
const CODER_NAMES = ['zlib', 'zstd', 'ANS']
const CODER_VARS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']

const LABEL_W = 100
const RIGHT_PAD = 64
const AXIS_H = 26
/** Strip under the axis reserved for the two reference-line labels, so they
 * never sit on top of the bars or each other. */
const REF_BAND = 30
const GROUP_H = 20
const ROW_H = 24
const BAR_H = 16

type Metric = 'bits' | 'ratio'

/** One drawn bar: a measured codec size. */
interface Row {
  key: string
  label: string
  name: string
  note: string
  bytes: number
  bitsPerSample: number
  ratio: number
  color: string
}

interface Tip {
  x: number
  y: number
  row: Row
}

function axisTicks(max: number): number[] {
  const step = max > 24 ? 8 : max > 12 ? 4 : max > 6 ? 2 : max > 3 ? 1 : 0.5
  const out: number[] = []
  for (let v = 0; v <= max + 1e-9; v += step) out.push(v)
  return out
}

/** A bar whose data-end is rounded (4px) while the baseline end stays square. */
function barPath(x0: number, y: number, len: number, h: number): string {
  const r = Math.min(4, len)
  return `M${x0},${y} h${len - r} a${r},${r} 0 0 1 ${r},${r} v${h - 2 * r} a${r},${r} 0 0 1 ${-r},${r} h${-(len - r)} z`
}

/** A reference-line label: a short sample of the line's own stroke, then the
 * value in ink, flipped to end-anchored near the right edge. */
function RefLabel(props: {
  x: number
  y: number
  width: number
  stroke: string
  dash: string
  text: string
}) {
  const flip = props.x > props.width - 190
  const dir = flip ? -1 : 1
  const x0 = props.x + 6 * dir
  return (
    <g>
      <line
        x1={x0}
        x2={x0 + 16 * dir}
        y1={props.y - 4}
        y2={props.y - 4}
        stroke={props.stroke}
        strokeWidth={1.5}
        strokeDasharray={props.dash}
      />
      <text
        x={x0 + 20 * dir}
        y={props.y}
        textAnchor={flip ? 'end' : 'start'}
        className="bar-value"
        fill="var(--ink)"
      >
        {props.text}
      </text>
    </g>
  )
}

interface Group {
  label: string
  rows: Row[]
}

/** The three prefilter groups of the Section 5.3 benchmark. */
function buildGroups(results: CodecResult[]): Group[] {
  return GROUPS.map((group, g) => ({
    label: group,
    rows: CODER_NAMES.map((coder, c) => ({ coder, c, r: results[g * 3 + c] }))
      .filter(x => x.r)
      .map(({ coder, c, r }) => ({
        key: r.codec,
        label: coder,
        name: r.codec,
        note: r.note,
        bytes: r.bytes,
        bitsPerSample: r.bitsPerSample,
        ratio: r.ratio,
        color: CODER_VARS[c],
      })),
  }))
}

export default function CompressionChart(props: {
  results: CodecResult[]
  /** The particle-filter estimate, once at least one replicate is in. */
  rateBits: number | null
  /** Standard error of that estimate, drawn as a band around its line. */
  rateSe: number | null
  /** The Section 3 spectral approximation, always shown dotted. */
  approxBits: number
  computing: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const width = useWidth(ref, 720)
  const [metric, setMetric] = useState<Metric>('ratio')
  const [tip, setTip] = useState<Tip | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)

  const { results, rateBits, rateSe, approxBits } = props
  if (results.length === 0) {
    return <p className="card-note">Computing compression on the first block…</p>
  }

  const groups = buildGroups(results)
  // Every coder that codes against an explicit model can be scored against it.
  const efficiency = results.filter(r => r.modelBitsPerSample !== undefined)

  // Group headers and bar rows laid out top to bottom; groups may differ in
  // row count, so positions accumulate rather than being indexed.
  const groupLabels: { label: string; y: number }[] = []
  const placed: { row: Row; y: number }[] = []
  let yCursor = AXIS_H + REF_BAND
  for (const g of groups) {
    groupLabels.push({ label: g.label, y: yCursor + 15 })
    yCursor += GROUP_H
    for (const row of g.rows) {
      placed.push({ row, y: yCursor })
      yCursor += ROW_H
    }
  }

  const rows = placed.map(p => p.row)
  const value = (r: { bitsPerSample: number; ratio: number }) =>
    metric === 'bits' ? r.bitsPerSample : r.ratio
  const rateValue =
    rateBits !== null && rateBits > 0 ? (metric === 'bits' ? rateBits : 16 / rateBits) : null
  const approxValue = approxBits > 0 ? (metric === 'bits' ? approxBits : 16 / approxBits) : null
  const xMax = Math.max(...rows.map(value), rateValue ?? 0, approxValue ?? 0) * 1.1

  const plotW = width - LABEL_W - RIGHT_PAD
  const height = yCursor + 6
  const xOf = (v: number) => LABEL_W + (v / xMax) * plotW

  const fmt = (r: Row) => (metric === 'bits' ? r.bitsPerSample.toFixed(2) : `${r.ratio.toFixed(2)}×`)

  const onBarMove = (e: React.PointerEvent, row: Row) => {
    const box = ref.current!.getBoundingClientRect()
    setTip({ x: e.clientX - box.left, y: e.clientY - box.top, row })
  }

  const rateX = rateValue !== null ? xOf(rateValue) : 0
  const rateLabel =
    rateBits !== null && rateBits > 0
      ? metric === 'bits'
        ? `particle filter = ${rateBits.toFixed(2)}`
        : `particle filter ⇒ ${(16 / rateBits).toFixed(2)}×`
      : ''
  const approxX = approxValue !== null ? xOf(approxValue) : 0
  const approxLabel =
    approxValue !== null
      ? metric === 'bits'
        ? `approx ≈ ${approxBits.toFixed(2)}`
        : `approx ⇒ ${(16 / approxBits).toFixed(2)}×`
      : ''
  // ± one standard error around the particle-filter line, in the plotted metric.
  let band: { x: number; w: number } | null = null
  if (rateBits !== null && rateBits > 0 && rateSe !== null && rateSe > 0) {
    const loBits = Math.max(rateBits - rateSe, 1e-9)
    const hiBits = rateBits + rateSe
    const x1 = xOf(metric === 'bits' ? loBits : 16 / hiBits)
    const x2 = Math.min(xOf(metric === 'bits' ? hiBits : 16 / loBits), LABEL_W + plotW)
    band = { x: x1, w: Math.max(x2 - x1, 0) }
  }

  return (
    <div>
      <div className="chart-header">
        <div>
          <div className="segmented" role="group" aria-label="metric">
            <button className={metric === 'ratio' ? 'active' : ''} onClick={() => setMetric('ratio')}>
              compression ratio
            </button>
            <button className={metric === 'bits' ? 'active' : ''} onClick={() => setMetric('bits')}>
              bits / sample
            </button>
          </div>
          <span className="metric-hint">
            {metric === 'bits' ? 'lower is better' : 'vs int16 — higher is better'}
          </span>
        </div>
        <div className="legend">
          {CODER_NAMES.map((name, i) => (
            <span key={name}>
              <span className="swatch" style={{ background: CODER_VARS[i] }} />
              {name}
            </span>
          ))}
        </div>
      </div>
      <div className={`chart-body${props.computing ? ' computing' : ''}`} ref={ref}>
        {props.computing && <span className="computing-badge">computing…</span>}
        <svg width={width} height={height}>
          {axisTicks(xMax).map(v => (
            <g key={v}>
              <line x1={xOf(v)} x2={xOf(v)} y1={AXIS_H - 6} y2={height - 4} stroke="var(--grid)" strokeWidth={1} />
              <text x={xOf(v)} y={AXIS_H - 10} textAnchor="middle" className="axis-tick">
                {+v.toFixed(1)}
              </text>
            </g>
          ))}
          <line x1={xOf(0)} x2={xOf(0)} y1={AXIS_H - 6} y2={height - 4} stroke="var(--baseline)" strokeWidth={1} />
          {band && (
            <rect
              x={band.x}
              y={AXIS_H - 2}
              width={band.w}
              height={height - 2 - AXIS_H}
              fill="var(--ink-2)"
              opacity={0.15}
            />
          )}
          {groupLabels.map(g => (
            <text key={g.label} x={0} y={g.y} className="bar-group-label">
              {g.label}
            </text>
          ))}
          {placed.map(({ row: r, y }) => {
            const len = Math.max(1, (value(r) / xMax) * plotW)
            return (
              <g key={r.key} opacity={hovered === null || hovered === r.key ? 1 : 0.45}>
                <text x={8} y={y + BAR_H / 2 + 4} className="bar-row-label">
                  {r.label}
                </text>
                <path d={barPath(xOf(0), y, len, BAR_H)} fill={r.color} />
                <text x={xOf(0) + len + 6} y={y + BAR_H / 2 + 4} className="bar-value">
                  {fmt(r)}
                </text>
                <rect
                  x={0}
                  y={y - (ROW_H - BAR_H) / 2}
                  width={width}
                  height={ROW_H}
                  fill="transparent"
                  onPointerMove={e => {
                    setHovered(r.key)
                    onBarMove(e, r)
                  }}
                  onPointerLeave={() => {
                    setHovered(null)
                    setTip(null)
                  }}
                />
              </g>
            )
          })}
          {approxValue !== null && (
            <g>
              <line
                x1={approxX}
                x2={approxX}
                y1={AXIS_H - 2}
                y2={height - 4}
                stroke="var(--theory)"
                strokeWidth={1.5}
                strokeDasharray="2 3"
              />
              <RefLabel
                x={approxX}
                y={AXIS_H + 11}
                width={width}
                stroke="var(--theory)"
                dash="2 3"
                text={approxLabel}
              />
            </g>
          )}
          {rateValue !== null && (
            <g>
              <line
                x1={rateX}
                x2={rateX}
                y1={AXIS_H - 2}
                y2={height - 4}
                stroke="var(--ink-2)"
                strokeWidth={1.5}
                strokeDasharray="5 4"
              />
              <RefLabel
                x={rateX}
                y={AXIS_H + 25}
                width={width}
                stroke="var(--ink-2)"
                dash="5 4"
                text={rateLabel}
              />
            </g>
          )}
        </svg>
        {tip && (
          <div className="viz-tooltip" style={{ left: tip.x + 14, top: tip.y - 8 }}>
            <div>
              <span className="tip-value">
                {tip.row.bitsPerSample.toFixed(3)} bits/sample · {tip.row.ratio.toFixed(2)}×
              </span>{' '}
              <span className="tip-label">{tip.row.name}</span>
            </div>
            <div className="tip-label">
              {tip.row.bytes.toLocaleString()} bytes · {tip.row.note}
            </div>
          </div>
        )}
      </div>
      {/* How much each entropy coder loses against the model it is coding
          against — its own overhead, separate from how good the model is. */}
      {efficiency.length > 0 && (
        <div className="coder-efficiency">
          <span className="coder-efficiency-title">
            entropy-coder overhead — output vs the order-0 model it codes against
          </span>
          {efficiency.map(r => {
            const model = r.modelBitsPerSample as number
            const over = model > 0 ? (r.bitsPerSample / model - 1) * 100 : 0
            return (
              <span key={r.codec} className="coder-efficiency-item">
                <span className="coder-efficiency-name">{r.codec}</span>
                <span className="coder-efficiency-bits">
                  {r.bitsPerSample.toFixed(3)} / {model.toFixed(3)} bits
                </span>
                <span className="coder-efficiency-pct">
                  {over >= 0 ? '+' : ''}
                  {over.toFixed(1)}%
                </span>
              </span>
            )
          })}
        </div>
      )}
      <details className="chart-table">
        <summary>Table view</summary>
        <table>
          <thead>
            <tr>
              <th>method</th>
              <th>bytes</th>
              <th>bits/sample</th>
              <th>ratio vs int16</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.key}>
                <td>{r.name}</td>
                <td>{r.bytes.toLocaleString()}</td>
                <td>{r.bitsPerSample.toFixed(3)}</td>
                <td>{r.ratio.toFixed(3)}</td>
              </tr>
            ))}
            {approxBits > 0 && (
              <tr>
                <td>entropy rate R̄ — spectral approximation</td>
                <td>—</td>
                <td>{approxBits.toFixed(3)}</td>
                <td>{(16 / approxBits).toFixed(3)}</td>
              </tr>
            )}
            {rateBits !== null && rateBits > 0 && (
              <tr>
                <td>entropy rate R̄ — particle filter</td>
                <td>—</td>
                <td>
                  {rateBits.toFixed(3)}
                  {rateSe !== null && rateSe > 0 ? ` ± ${rateSe.toFixed(3)}` : ''}
                </td>
                <td>{(16 / rateBits).toFixed(3)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </details>
    </div>
  )
}
