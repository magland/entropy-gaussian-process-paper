from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_toeplitz

from .process import generate, quantize

INT16_MIN, INT16_MAX = -32768, 32767


def quantized_sample(kernel, delta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    y = quantize(generate(kernel, int(n), rng), delta)
    if y.min() < INT16_MIN or y.max() > INT16_MAX:
        raise ValueError(
            f"quantized values span [{y.min()}, {y.max()}], outside int16; "
            "the 16-bit baseline does not apply at this delta"
        )
    return y


@dataclass
class LPCModel:
    coefs: np.ndarray  # coefs / 2**shift approximates the predictor
    shift: int
    order: int

    @property
    def header_bytes(self) -> int:
        return 2 * self.order + 1


def lpc_transform(y, order: int, ridge: float = 1e-9):
    """Fixed-point linear prediction, FLAC style.

    The leading `order` samples pass through unpredicted, so the residual has
    the same length and the transform inverts exactly in integer arithmetic.
    """
    y = np.asarray(y, dtype=np.int64)
    n = y.size
    order = int(min(order, n // 2))
    if order < 1:
        return y.copy(), LPCModel(np.zeros(0, dtype=np.int64), 0, 0)

    spec = np.fft.rfft(y.astype(float), 2 * n)
    ac = np.fft.irfft(spec * spec.conj())[: order + 1]
    ac[0] = ac[0] * (1.0 + ridge) + ridge
    a = solve_toeplitz(ac[:order], ac[1 : order + 1])

    shift = int(np.clip(np.floor(np.log2(INT16_MAX / max(np.abs(a).max(), 1e-12))), 0, 15))
    q = np.round(a * (1 << shift)).astype(np.int64)
    pred = np.zeros(n, dtype=np.int64)
    for i in range(order):
        pred[order:] += q[i] * y[order - 1 - i : n - 1 - i]
    residual = y.copy()
    residual[order:] -= pred[order:] >> shift
    return residual, LPCModel(q, shift, order)


def lpc_inverse(residual, model: LPCModel) -> np.ndarray:
    """Reconstruct one sample at a time, as a decoder must."""
    y = np.asarray(residual, dtype=np.int64).copy()
    q, shift, order = model.coefs, model.shift, model.order
    for t in range(order, y.size):
        y[t] += int(q @ y[t - order : t][::-1]) >> shift
    return y


def delta_transform(y) -> np.ndarray:
    """First difference of the integer stream, the order-1 predictor with a
    fixed coefficient.  The leading sample passes through, so the transform
    inverts exactly by cumulative sum."""
    y = np.asarray(y, dtype=np.int64)
    d = y.copy()
    d[1:] -= y[:-1]
    return d


def delta_inverse(d) -> np.ndarray:
    return np.cumsum(np.asarray(d, dtype=np.int64))


def _as_int16(a) -> np.ndarray:
    """int16 view of an integer array, refusing values the type cannot hold.

    A silent wrap would make the reported size meaningless, so the codec is
    recorded as failed instead.
    """
    a = np.asarray(a)
    if a.size and (a.min() < INT16_MIN or a.max() > INT16_MAX):
        raise ValueError(
            f"values span [{a.min()}, {a.max()}], outside int16; "
            "the 16-bit serialization does not apply"
        )
    return a.astype(np.int16)


def _int16_bytes(a) -> bytes:
    return _as_int16(a).astype("<i2").tobytes()


#: Nominal sample rate written into the FLAC header; it affects the encoded
#: size only through the fixed metadata block.
FLAC_RATE = 30_000


def flac_size(a, level: int = 8) -> int:
    """Bytes of the int16 stream encoded by the reference FLAC encoder.

    Uses pyFLAC when it is installed and libsndfile through soundfile
    otherwise; both drive libFLAC and were checked to produce byte-identical
    streams.  The size includes FLAC's own header, about 50 bytes.
    """
    samples = _as_int16(a)
    try:
        import pyflac
    except ImportError:
        pass
    else:
        chunks: list[int] = []
        encoder = pyflac.StreamEncoder(
            write_callback=lambda buf, nbytes, nsamples, current: chunks.append(len(buf)),
            sample_rate=FLAC_RATE,
            compression_level=level,
        )
        encoder.process(samples.reshape(-1, 1))
        encoder.finish()
        return int(sum(chunks))

    import io

    import soundfile as sf

    buf = io.BytesIO()
    with sf.SoundFile(
        buf, "w", samplerate=FLAC_RATE, channels=1, format="FLAC", subtype="PCM_16"
    ) as f:
        # libsndfile takes the compression level as a fraction of the maximum,
        # through a command not exposed by soundfile's public interface.
        value = sf._ffi.new("double*", level / 8.0)
        if sf._snd.sf_command(f._file, 0x1301, value, sf._ffi.sizeof("double")) != 1:
            raise RuntimeError("libsndfile refused the FLAC compression level")
        f.write(samples)
    return len(buf.getvalue())


def codec_registry():
    """name -> callable(integer array) -> compressed size in bytes.

    Byte-oriented codecs see the little-endian int16 serialization; ANS codes
    the integers directly and its size includes the symbol table; FLAC encodes
    the stream as single-channel 16-bit audio.
    """
    import bz2
    import lzma
    import zlib

    reg = {
        "zlib-9": lambda a: len(zlib.compress(_int16_bytes(a), 9)),
        "bz2-9": lambda a: len(bz2.compress(_int16_bytes(a), 9)),
        "lzma-9": lambda a: len(lzma.compress(_int16_bytes(a), preset=9)),
    }
    try:
        import zstandard

        reg["zstd-19"] = lambda a: len(
            zstandard.ZstdCompressor(level=19).compress(_int16_bytes(a))
        )
    except ImportError:
        pass
    try:
        import brotli

        reg["brotli-11"] = lambda a: len(brotli.compress(_int16_bytes(a), quality=11))
    except ImportError:
        pass
    try:
        import lz4.frame

        reg["lz4-12"] = lambda a: len(
            lz4.frame.compress(_int16_bytes(a), compression_level=12)
        )
    except ImportError:
        pass
    try:
        from simple_ans import ans_encode

        reg["ans"] = lambda a: ans_encode(_as_int16(a)).size()
    except ImportError:
        pass
    try:
        import pyflac  # noqa: F401
    except ImportError:
        try:
            import soundfile  # noqa: F401
        except ImportError:
            return reg
    reg["flac-8"] = lambda a: flac_size(a, 8)
    return reg


@dataclass
class CodecResult:
    method: str
    transform: str
    n_bytes: int
    bits_per_sample: float
    ratio: float  # against the 16-bit baseline
    seconds: float
    error: str = ""


def empirical_entropy_bits(a) -> float:
    _, counts = np.unique(np.asarray(a), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def compress_benchmark(y, *, lpc_order: int = 0, transforms=None, methods=None, progress=None):
    """Every available codec on every requested transform of the stream,
    sorted by bits per sample, best first.

    The transforms are ``raw``, ``delta`` (first difference), and ``lpc``
    (fixed-point linear prediction at ``lpc_order``); by default the LPC stage
    is included whenever an order is given.  Each is integer-reversible, so
    the sizes are those of a lossless coder.
    """
    y = np.asarray(y, dtype=np.int64)
    n = y.size
    if transforms is None:
        transforms = ("raw", "lpc") if lpc_order else ("raw",)
    stages = []
    for name in transforms:
        if name == "raw":
            stages.append(("raw", y, 0))
        elif name == "delta":
            stages.append(("delta", delta_transform(y), 0))
        elif name == "lpc":
            if not lpc_order:
                raise ValueError("the lpc transform requires lpc_order > 0")
            residual, model = lpc_transform(y, lpc_order)
            if model.order:
                stages.append((f"lpc({model.order})", residual, model.header_bytes))
        else:
            raise ValueError(f"unknown transform {name!r}; choose from raw, delta, lpc")

    reg = codec_registry()
    if methods:
        missing = [x for x in methods if x not in reg]
        if missing:
            raise ValueError(f"unknown or unavailable codecs: {missing}")
        reg = {k: reg[k] for k in methods}

    out = []
    for transform, data, header in stages:
        for name, fn in reg.items():
            t0 = time.monotonic()
            try:
                n_bytes = fn(data) + header
                bits = 8.0 * n_bytes / n
                res = CodecResult(
                    name, transform, n_bytes, bits, 16.0 / bits, time.monotonic() - t0
                )
            except Exception as exc:  # a codec's limits are not a run-ending error
                res = CodecResult(
                    name, transform, 0, float("nan"), float("nan"),
                    time.monotonic() - t0, error=str(exc),
                )
            out.append(res)
            if progress is not None:
                progress("codec", {"result": res})
    return sorted(out, key=lambda r: (np.isnan(r.bits_per_sample), r.bits_per_sample))
