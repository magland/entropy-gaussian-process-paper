import {
  FAMILY_LABELS,
  SIGMA_STOPS,
  TAP_STOPS,
  WIDTH_STOPS,
  frequencyStops,
  maxCutoffHz,
  nearestStop,
  type FilterFamily,
  type FilterSpec,
} from '../model/filters'

export interface ControlsProps {
  sigma: number
  setSigma: (v: number) => void
  sampleRateHz: number
  setSampleRateHz: (v: number) => void
  spec: FilterSpec
  setSpec: (v: FilterSpec) => void
}

function formatHz(hz: number): string {
  return hz >= 1000 ? `${+(hz / 1000).toFixed(1)} kHz` : `${hz} Hz`
}

/** A slider that moves between the given values rather than a continuum. */
function StopSlider(props: {
  label: string
  stops: number[]
  value: number
  format: (v: number) => string
  onChange: (v: number) => void
}) {
  const { stops } = props
  const snapped = nearestStop(stops, props.value)
  const index = stops.indexOf(snapped)
  return (
    <div className="control">
      <label>
        {props.label} <span className="value">{props.format(snapped)}</span>
      </label>
      <input
        type="range"
        min={0}
        max={stops.length - 1}
        step={1}
        value={index}
        onChange={e => props.onChange(stops[Number(e.target.value)])}
      />
    </div>
  )
}

export default function Controls(p: ControlsProps) {
  const { spec } = p
  const freqStops = frequencyStops(maxCutoffHz(p.sampleRateHz))
  const sinc = spec.family === 'lowpass' || spec.family === 'bandpass'
  return (
    <div className="controls">
      <StopSlider
        label="σ (quantization steps)"
        stops={SIGMA_STOPS}
        value={p.sigma}
        format={v => String(v)}
        onChange={p.setSigma}
      />
      <div className="control">
        <label>filter</label>
        <select
          value={spec.family}
          onChange={e => p.setSpec({ ...spec, family: e.target.value as FilterFamily })}
        >
          {(Object.keys(FAMILY_LABELS) as FilterFamily[]).map(f => (
            <option key={f} value={f}>
              {FAMILY_LABELS[f]}
            </option>
          ))}
        </select>
      </div>
      {spec.family === 'bandpass' && (
        <StopSlider
          label="low cutoff"
          stops={freqStops.filter(f => f < spec.highHz)}
          value={spec.lowHz}
          format={formatHz}
          onChange={v => p.setSpec({ ...spec, lowHz: v })}
        />
      )}
      {sinc && (
        <StopSlider
          label={spec.family === 'bandpass' ? 'high cutoff' : 'cutoff'}
          stops={spec.family === 'bandpass' ? freqStops.filter(f => f > spec.lowHz) : freqStops}
          value={spec.highHz}
          format={formatHz}
          onChange={v => p.setSpec({ ...spec, highHz: v })}
        />
      )}
      {sinc && (
        <StopSlider
          label="kernel length"
          stops={TAP_STOPS}
          value={spec.taps}
          format={v => `${v} taps`}
          onChange={v => p.setSpec({ ...spec, taps: v })}
        />
      )}
      {spec.family === 'movingAverage' && (
        <StopSlider
          label="width"
          stops={WIDTH_STOPS}
          value={spec.width}
          format={v => `${v} samples`}
          onChange={v => p.setSpec({ ...spec, width: v })}
        />
      )}
      <div className="control">
        <label>sample rate</label>
        <select value={p.sampleRateHz} onChange={e => p.setSampleRateHz(Number(e.target.value))}>
          {[1000, 8000, 16000, 30000, 44100, 96000].map(r => (
            <option key={r} value={r}>
              {formatHz(r)}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
