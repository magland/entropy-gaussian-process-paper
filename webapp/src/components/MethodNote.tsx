import { useMemo } from 'react'
import katex from 'katex'

function Tex({ tex, display }: { tex: string; display?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: !!display, throwOnError: false }),
    [tex, display],
  )
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

/** Where the entropy rate R̄ comes from, in brief. */
export default function MethodNote() {
  return (
    <div className="math-section">
      <p>
        R̄ is the entropy rate of the quantized process z — the bits per sample no lossless code
        can beat. It has no usable closed form, but the Shannon–McMillan–Breiman theorem turns it
        into a likelihood computation on <em>one long typical sequence</em> drawn from the process
        itself:
      </p>
      <Tex display tex="\bar H \;=\; \lim_{n\to\infty} -\tfrac1n \log_2 P(z_1,\dots,z_n) \quad \text{a.s.}" />
      <p>
        That likelihood is an n-dimensional Gaussian integral over a box — intractable directly,
        tractable sequentially. The process is written in FIR form,{' '}
        <Tex tex="x_t = \sum_{j<L} \tilde h_j\, w_{t-j}" /> with w i.i.d. standard normal, using
        the <em>minimum-phase</em> spectral factor h̃ (computed by the cepstral construction shown
        in the filter section). That makes z a hidden Markov process whose state is the last L−1
        innovations, and the structure is exactly solvable one step at a time: given the state,
        the next sample is <Tex tex="\mathcal N(\mu, h_0^2)" />, so the probability of the
        observed cell is <Tex tex="\Phi(r) - \Phi(l)" /> and the innovation conditioned on the
        cell is a truncated normal. A particle filter with these exact ingredients — the fully
        adapted filter — estimates each predictive term, and
      </p>
      <Tex display tex="\hat{\bar H} \;=\; -\frac{1}{n} \sum_{t=1}^{n} \log_2 \hat P\big(z_t \mid z_1^{t-1}\big)" />
      <p>
        is consistent as n and the particle count N grow, approaching R̄ from above (the particle
        likelihood is unbiased, so Jensen's inequality biases the log upward by O(1/N) per
        sample). Each replicate draws a fresh sequence and a fresh filter, so the spread across
        replicates is an honest standard error. The number that governs difficulty is Δ/h₀ —
        quantization step over one-step prediction error. When it is small, the observations
        nearly determine the state and too few particles make the cloud lose it entirely; the
        estimate is then meaningless rather than noisy, and the effective sample size does not
        detect it (the surviving particles agree with each other, not with the data). What does
        detect it is the per-step log-likelihood falling ~25 nats below{' '}
        <Tex tex="\log\min(1, \Delta/h_0)" /> — reported here as "lost lock". Note the scale
        term: under fine quantization every step is legitimately worth about log(Δ/h₀) nats, so a
        fixed threshold would misfire.
      </p>
      <p>
        The dotted line is the analytic spectral approximation — the classical high-resolution
        formula with the quantizer's noise power Δ²/12 filling in the spectral zeros,
      </p>
      <Tex display tex="\bar H \;\approx\; \frac{1}{4\pi}\int_{-\pi}^{\pi} \log_2 \frac{2\pi e\,\big(S(\omega) + \Delta^2/12\big)}{\Delta^2}\, d\omega," />
      <p>
        computed in milliseconds from the spectrum alone. It shares nothing with the filter — no
        factorization, no sampling — which makes it a genuinely independent second opinion:
        within ~0.1 bit of the filter for Δ ≲ σ, and the sharp reference to check against when
        the filter collapses. Its one structural failure is coarse quantization, where it tends
        to a 0.25-bit floor while the true rate tends to zero — the additive-noise model's
        artifact, since rounding a signal much smaller than Δ produces a constant symbol rather
        than Δ²/12 of noise.
      </p>
      <p className="card-note">
        The estimate button runs exactly this method in a web worker — a TypeScript port of the
        companion egp Python package (src/egp, hand-synced) — until stopped. Two engines: a
        scalar one, one replicate at a time, and WebGPU (the default where available), which
        runs each replicate as one workgroup — 256 threads across the particles, with the
        per-step reductions in workgroup memory — and several replicates as concurrent
        workgroups, in f32 log space with the same lost-lock diagnostics. The command line runs
        the Python original at the same settings as an independent check; the port is validated
        against it by the golden tests in test/.
      </p>
    </div>
  )
}
