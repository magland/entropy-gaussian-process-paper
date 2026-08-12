"""GPU implementation of the fully adapted particle filter, via WebGPU.

Runs the WGSL compute shader in ``pf.wgsl`` — the same shader the companion
webapp executes in the browser — through `wgpu-py <https://github.com/pygfx/
wgpu-py>`_, which reaches any Vulkan/Metal/D3D12 device, including integrated
GPUs.  Same algorithm and diagnostics as :mod:`.pf`: one workgroup of 256
threads runs one replicate (the phases of a step are sequential, so a single
replicate can only use one workgroup's worth of parallelism), and a batch of
replicates runs as concurrent workgroups of one dispatch — which is where the
device gets filled and the speedup comes from.

The shader works in f32: log Phi through an Abramowitz-Stegun erf and the
Laplace continued fraction for the Mills ratio; its inverse seeded by Acklam's
branches with the tails computed from log p directly (where exp would cancel)
and polished by Newton.  Weight errors are ~1e-4 relative and draw errors
~1e-2 at 5 sigma — far below the estimator's statistical resolution; the
per-replicate estimates agree with :func:`.pf.particle_filter_loglik` to well
within one Monte-Carlo standard error, with identical ESS.  Per-step
log-likelihoods are read back as f32 and accumulated in f64 here, where the
lost-lock diagnostic is evaluated exactly as in the CPU filter.

``wgpu`` is an optional dependency: ``pip install egp[gpu]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pf import DEFAULT_COLLAPSE_MARGIN, FilterResult, collapse_threshold

_SHADER_PATH = Path(__file__).with_name("pf.wgsl")

_device_cache: list = []  # [(device, pipeline)] once created


def gpu_available() -> bool:
    """True when wgpu is installed and a WebGPU adapter answers."""
    try:
        import wgpu
    except ImportError:
        return False
    try:
        return wgpu.gpu.request_adapter_sync(power_preference="high-performance") is not None
    except Exception:
        return False


def gpu_description() -> str:
    """Human-readable name of the adapter the filter would run on."""
    import wgpu

    info = wgpu.gpu.request_adapter_sync(power_preference="high-performance").info
    return f"{info.get('device', '?')} ({info.get('backend_type', '?')})"


def _get_device():
    """The process-wide device and compiled pipeline (compilation is ~100 ms)."""
    if _device_cache:
        return _device_cache[0]
    try:
        import wgpu
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            'the GPU engine needs the wgpu package: pip install "egp[gpu]"'
        ) from exc
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:  # pragma: no cover - depends on environment
        raise RuntimeError("no WebGPU adapter found; use the CPU engine")
    device = adapter.request_device_sync()
    module = device.create_shader_module(code=_SHADER_PATH.read_text())
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"}
    )
    _device_cache.append((device, pipeline))
    return _device_cache[0]


def max_batch(n_particles: int, n_taps: int, n: int) -> int:
    """Replicates one dispatch can hold within the device's buffer limits."""
    device, _ = _get_device()
    limit = device.limits["max-storage-buffer-binding-size"]
    ring_len = max(n_taps - 1, 1)
    per_rep = max(n_particles * ring_len * 4, n_particles * 16, n * 8)
    fit = limit // per_rep
    if fit < 1:
        raise RuntimeError(
            f"one replicate needs {per_rep / 1e6:.0f} MB of GPU buffer, over the "
            f"device limit of {limit / 1e6:.0f} MB; reduce n_particles or use the CPU engine"
        )
    return int(fit)


def particle_filter_loglik_gpu(
    ys,
    taps,
    delta: float,
    n_particles: int,
    seeds,
    collapse_margin: float = DEFAULT_COLLAPSE_MARGIN,
    burn_in: int = 0,
    progress=None,
) -> list[FilterResult]:
    """Run one GPU batch: one particle filter per row of ``ys``, concurrently.

    Parameters mirror :func:`.pf.particle_filter_loglik` except that ``ys`` is
    a list/array of R equal-length symbol sequences and ``seeds`` is one u32
    per replicate for the shader's counter-based RNG (the observed sequences
    come from the caller, so the engines can be fed identical data).
    ``progress(done, total)`` is called as the step sweep advances.
    """
    device, pipeline = _get_device()
    import wgpu

    y_batch = np.ascontiguousarray(ys, dtype=np.int32)
    if y_batch.ndim != 2:
        raise ValueError("ys must be R sequences of equal length")
    R, n = y_batch.shape
    taps = np.asarray(taps, dtype=float).ravel()
    L = taps.size
    ring_len = L - 1
    h0 = float(taps[0])
    if ring_len < 1:
        raise ValueError("the GPU path needs L >= 2; the white process is exact on the CPU")
    if not h0 > 0:
        raise ValueError("taps[0] must be positive (use the minimum-phase factorization)")
    seeds = np.ascontiguousarray(seeds, dtype=np.uint32)
    if seeds.size != R:
        raise ValueError("one seed per replicate required")
    N = int(n_particles)
    if N < 2:
        raise ValueError("n_particles must be >= 2")
    burn_in = int(np.clip(burn_in, 0, max(0, n - 1)))
    if R > max_batch(N, L, n):
        raise RuntimeError(f"batch of {R} replicates exceeds GPU buffer limits")

    usage = wgpu.BufferUsage
    storage = usage.STORAGE | usage.COPY_DST

    def buf(size, u=storage):
        return device.create_buffer(size=max(int(size), 16), usage=u)

    buffers = {
        "uniform": buf(48, usage.UNIFORM | usage.COPY_DST),
        "ringA": buf(R * N * ring_len * 4),
        "ringB": buf(R * N * ring_len * 4),
        "ys": buf(R * n * 4),
        "stepOut": buf(R * n * 8, storage | usage.COPY_SRC),
        "scratch": buf(R * N * 16),
        "cum": buf(R * N * 4),
        "taps": buf(L * 4),
        "repSeeds": buf(R * 4),
    }
    order = ["uniform", "ringA", "ringB", "ys", "stepOut", "scratch", "cum", "taps", "repSeeds"]
    try:
        device.queue.write_buffer(buffers["ys"], 0, y_batch)
        device.queue.write_buffer(buffers["taps"], 0, taps.astype(np.float32))
        device.queue.write_buffer(buffers["repSeeds"], 0, seeds)
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {
                    "binding": i,
                    "resource": {"buffer": buffers[k], "offset": 0, "size": buffers[k].size},
                }
                for i, k in enumerate(order)
            ],
        )

        uniform = np.zeros(12, dtype=np.uint32)
        f32 = uniform.view(np.float32)
        uniform[0], uniform[1], uniform[2] = N, ring_len, n
        f32[8], f32[9], f32[10], f32[11] = h0, delta / h0, delta, np.log(N)

        # Steps per submit, sized to keep one dispatch well under any device
        # timeout; syncing (a tiny readback) every few chunks bounds how much
        # work is in flight and gives the progress callback real meaning.
        chunk = max(8, min(512, int(3e7 / (N * (ring_len + 30)))))
        chunks = range(0, n, chunk)
        sync_every = max(1, len(chunks) // 50)
        for ci, start in enumerate(chunks):
            uniform[3], uniform[4] = start, min(chunk, n - start)
            device.queue.write_buffer(buffers["uniform"], 0, uniform)
            encoder = device.create_command_encoder()
            comp = encoder.begin_compute_pass()
            comp.set_pipeline(pipeline)
            comp.set_bind_group(0, bind_group)
            comp.dispatch_workgroups(R)
            comp.end()
            device.queue.submit([encoder.finish()])
            if ci % sync_every == sync_every - 1:
                device.queue.read_buffer(buffers["stepOut"], 0, size=8)  # sync point
                if progress is not None:
                    progress(min(start + chunk, n), n)

        raw = device.queue.read_buffer(buffers["stepOut"])
        if progress is not None:
            progress(n, n)
    finally:
        for b in buffers.values():
            b.destroy()

    # Per-step values accumulate in f64 here, where the lost-lock diagnostic
    # is evaluated exactly as in the CPU filter.
    steps = np.frombuffer(raw, dtype=np.float32).reshape(R, n, 2).astype(np.float64)
    limit = collapse_threshold(delta, h0, collapse_margin)
    results = []
    for r in range(R):
        loglik = steps[r, :, 0]
        ess = steps[r, :, 1]
        lost = loglik < limit
        results.append(
            FilterResult(
                loglik_nats=float(loglik.sum()),
                loglik_counted=float(loglik[burn_in:].sum()),
                n_steps=n,
                burn_in=burn_in,
                mean_ess=float(ess.mean()),
                min_ess=float(ess.min()),
                n_lost_lock=int(lost.sum()),
                first_lost_step=int(np.argmax(lost)) if lost.any() else -1,
                collapse_threshold=limit,
            )
        )
    return results
