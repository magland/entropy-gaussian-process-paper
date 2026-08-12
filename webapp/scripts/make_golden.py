"""Regenerate golden.json from the egp Python reference implementation.

The TypeScript code in src/egp is a hand-synced port of ../egp; these values
pin the port to the original. Run from the webapp directory with the egp
package importable (pip install -e ../egp):

    python scripts/make_golden.py
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "egp" / "src"))

from scipy.special import log_ndtr, ndtri_exp  # noqa: E402

import egp  # noqa: E402
from egp.approx import entropy_rate_approx  # noqa: E402
from egp.estimate import fine_quantization_bits, marginal_entropy_bits  # noqa: E402
from egp.factor import minimum_phase_from_taps  # noqa: E402
from egp.pf import log_cell_prob  # noqa: E402


def hamming_lowpass(fc: float, taps: int) -> np.ndarray:
    """The app's windowed-sinc lowpass (model/filters.ts), unit DC gain."""
    n = taps | 1
    mid = (n - 1) / 2
    i = np.arange(n)
    t = i - mid
    sinc = np.where(t == 0, 2 * fc, np.sin(2 * np.pi * fc * t) / (np.pi * np.where(t == 0, 1, t)))
    w = 0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))
    h = sinc * w
    return h / h.sum()


def bandpass(lo: float, hi: float, taps: int) -> np.ndarray:
    return hamming_lowpass(hi, taps) - hamming_lowpass(lo, taps)


KERNELS = {
    "white": np.array([1.0]),
    "ma8": np.full(8, 1 / 8),
    "diff": np.array([1.0, -1.0]),
    "lowpass_6k_30k_65": hamming_lowpass(6000 / 30000, 65),
    "bandpass_300_6000_30k_101": bandpass(300 / 30000, 6000 / 30000, 101),
}

SIGMAS = {"white": 5.0, "ma8": 5.0, "diff": 5.0, "lowpass_6k_30k_65": 20.0, "bandpass_300_6000_30k_101": 20.0}

out = {
    "logNdtr": {},
    "ndtriExp": {},
    "logCellProb": {},
    "kernels": {},
}

for x in [-200.0, -100.0, -50.0, -37.6, -37.0, -20.0, -10.0, -5.0, -1.0, -0.1, 0.0, 0.5, 1.0, 5.0, 10.0, 37.0]:
    out["logNdtr"][repr(x)] = float(log_ndtr(x))

for y in [-2e5, -1e4, -1000.0, -700.0, -650.0, -100.0, -10.0, -1.0, -0.1, -0.01, -1e-6]:
    out["ndtriExp"][repr(y)] = float(ndtri_exp(y))

for lo, hi in [(-0.5, 0.5), (0.0, 1.0), (-40.0, -39.0), (39.0, 40.0), (-3.0, 5.0), (100.0, 100.1)]:
    out["logCellProb"][f"{lo},{hi}"] = float(log_cell_prob(lo, hi))

for name, kernel in KERNELS.items():
    sigma = SIGMAS[name]
    taps = sigma * kernel
    fac = minimum_phase_from_taps(taps)
    sigma_y = float(np.sqrt(np.dot(taps, taps)))
    entry = {
        "sigma": sigma,
        "sigmaY": sigma_y,
        "h0": fac.h0,
        "minPhaseHead": [float(v) for v in fac.taps[:8]],
        "autocovError": fac.autocov_error,
        "flooredFraction": fac.floored_fraction,
        "approxBits": float(entropy_rate_approx(taps, 1.0)),
        "lowerBoundBits": max(0.0, float(fine_quantization_bits(fac.h0, 1.0))),
        "upperBoundBits": float(marginal_entropy_bits(sigma_y, 1.0)),
    }
    out["kernels"][name] = entry

# A reference particle-filter run for statistical (not bitwise) agreement: the
# RNGs differ across languages, so the check is mean ± a few standard errors.
spec = egp.from_preset("ma", taps=8, sigma=None)  # natural scale: ones(8)
spec.taps[:] = minimum_phase_from_taps(5.0 * KERNELS["ma8"]).taps
res = egp.estimate_entropy_rate(spec, delta=1.0, n=20000, n_particles=1000, n_repeats=5, seed=1)
out["pfReference"] = {
    "kernel": "ma8",
    "n": 20000,
    "nParticles": 1000,
    "meanBits": res.entropy_rate_bits,
    "stderrBits": res.stderr_bits,
    "meanEss": res.mean_ess,
}

path = pathlib.Path(__file__).resolve().parents[1] / "golden.json"
path.write_text(json.dumps(out, indent=2))
print(f"wrote {path}")
print(f"pf reference: {res.entropy_rate_bits:.4f} ± {res.stderr_bits:.4f} bits/sample")
