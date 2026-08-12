"""Entropy rate of a quantized stationary Gaussian process.

The process X_t is stationary Gaussian, written in FIR form
``X_t = sum_j h_j W_{t-j}`` with W i.i.d. standard normal, and is observed
through a uniform quantizer ``Y_t = round(X_t / delta)`` taking integer values.
The entropy rate, in bits per sample, is estimated by drawing one long typical
sequence and evaluating its log-likelihood with a fully adapted particle
filter (Shannon-McMillan-Breiman).

Typical use::

    import egp

    spec = egp.from_preset("lowpass", cutoff=6000, fs=30000, taps=65)
    result = egp.estimate_entropy_rate(spec, delta=0.5, n=100_000, n_particles=2000)
    print(result.entropy_rate_bits)
"""

from .approx import (
    QUANTIZER_FLOOR,
    entropy_rate_approx,
    entropy_rate_lower_bound,
    innovation_variance,
    psd,
    snr_integral,
)
from .compression import (
    CodecResult,
    LPCModel,
    codec_registry,
    compress_benchmark,
    empirical_entropy_bits,
    lpc_inverse,
    lpc_transform,
    quantized_sample,
)
from .estimate import (
    EntropyRateEstimate,
    RepeatUpdate,
    differential_entropy_bits,
    empirical_symbol_entropy_bits,
    estimate_entropy_rate,
    fine_quantization_bits,
    marginal_entropy_bits,
)
from .factor import (
    Factorization,
    autocovariance_from_taps,
    minimum_phase_from_autocov,
    minimum_phase_from_taps,
    spectrum_from_autocov,
    szego_h0,
)
from .pf import FilterResult, collapse_threshold, particle_filter_loglik
from .pf_gpu import gpu_available, particle_filter_loglik_gpu
from .process import dequantize, generate, generate_quantized, quantize
from .spec import PRESETS, ProcessSpec, from_autocov, from_preset, from_taps

__version__ = "0.1.0"

__all__ = [
    "CodecResult",
    "LPCModel",
    "QUANTIZER_FLOOR",
    "codec_registry",
    "compress_benchmark",
    "empirical_entropy_bits",
    "entropy_rate_approx",
    "entropy_rate_lower_bound",
    "innovation_variance",
    "lpc_inverse",
    "lpc_transform",
    "psd",
    "quantized_sample",
    "snr_integral",
    "PRESETS",
    "EntropyRateEstimate",
    "Factorization",
    "FilterResult",
    "ProcessSpec",
    "RepeatUpdate",
    "autocovariance_from_taps",
    "collapse_threshold",
    "dequantize",
    "differential_entropy_bits",
    "empirical_symbol_entropy_bits",
    "estimate_entropy_rate",
    "fine_quantization_bits",
    "from_autocov",
    "from_preset",
    "from_taps",
    "generate",
    "generate_quantized",
    "gpu_available",
    "marginal_entropy_bits",
    "minimum_phase_from_autocov",
    "minimum_phase_from_taps",
    "particle_filter_loglik",
    "particle_filter_loglik_gpu",
    "quantize",
    "spectrum_from_autocov",
    "szego_h0",
    "__version__",
]
