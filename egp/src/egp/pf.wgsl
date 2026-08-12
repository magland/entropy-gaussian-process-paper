// The fully adapted particle filter as a WGSL compute shader — the single
// source shared by the webapp (imported ?raw into src/egp/pfGpu.ts) and the
// Python package (loaded by egp/pf_gpu.py through wgpu-py). One workgroup of
// 256 threads runs one replicate, looping over the observation sequence with
// barriers between the phases of a step; replicates run as concurrent
// workgroups of one dispatch. See pfGpu.ts / pf_gpu.py for the buffer layout
// and the f32 accuracy notes behind the log-space normal functions here.

const WG = 256u;
const NEG_BIG = -1.0e30;
const HALF_LOG_2PI = 0.9189385332046727;

struct Params {
  nParticles: u32,
  ringLen: u32,
  nSteps: u32,
  chunkStart: u32,
  chunkLen: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  h0: f32,
  width: f32,
  delta: f32,
  logN: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> ringA: array<f32>;
@group(0) @binding(2) var<storage, read_write> ringB: array<f32>;
@group(0) @binding(3) var<storage, read> ys: array<i32>;
@group(0) @binding(4) var<storage, read_write> stepOut: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> scratch: array<vec4f>; // la, logAlpha, lo, mirror
@group(0) @binding(6) var<storage, read_write> cum: array<f32>;
@group(0) @binding(7) var<storage, read> taps: array<f32>;
@group(0) @binding(8) var<storage, read> repSeeds: array<u32>;

// ---- counter-based RNG ----------------------------------------------------

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn uni(seed: u32, stream: u32, ctr: u32) -> f32 {
  return (f32(pcg(seed ^ pcg(stream ^ pcg(ctr)))) + 0.5) / 4294967296.0;
}

fn normalDraw(seed: u32, stream: u32, ctr: u32) -> f32 {
  let u1 = max(uni(seed, stream, 2u * ctr), 1e-12);
  let u2 = uni(seed, stream, 2u * ctr + 1u);
  return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

// ---- f32 log-space normal functions ---------------------------------------

// Abramowitz–Stegun 7.1.26: erf for x >= 0, |abs err| < 1.5e-7.
fn erfPos(x: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * x);
  let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return 1.0 - poly * exp(-x * x);
}

fn phiHi(x: f32) -> f32 { // Phi(x) for x >= -1.5
  let z = x * 0.70710678;
  if (z >= 0.0) { return 0.5 * (1.0 + erfPos(z)); }
  return 0.5 * (1.0 - erfPos(-z));
}

// log Phi(x): erf branch above -1.5, Laplace continued fraction for the
// Mills ratio below (depth 16 is ~1e-4 relative at the branch point and
// improves rapidly further out).
fn logNdtr(x: f32) -> f32 {
  if (x >= -1.5) { return log(phiHi(x)); }
  let a = -x;
  var f = 0.0;
  for (var k: i32 = 16; k >= 1; k--) { f = f32(k) / (a + f); }
  return -0.5 * a * a - HALF_LOG_2PI - log(a + f);
}

fn log1mexp(x: f32) -> f32 { // log(1 - e^x) for x <= 0
  if (x > -0.6931472) { return log(max(-(exp(x) - 1.0), 1e-38)); }
  return log(1.0 - exp(x));
}

fn logaddexp(a: f32, b: f32) -> f32 {
  let m = max(a, b);
  return m + log(exp(a - m) + exp(b - m));
}

// Acklam tail branch, x for quantile q = sqrt(-2 ln p_tail).
fn acklamTail(q: f32) -> f32 {
  let num = ((((-7.784894002430293e-3 * q - 3.223964580411365e-1) * q - 2.400758277161838) * q - 2.549732539343734) * q + 4.374664141464968) * q + 2.938163982698783;
  let den = (((7.784695709041462e-3 * q + 3.224671290700398e-1) * q + 2.445134137142996) * q + 3.754408661907416) * q + 1.0;
  return num / den;
}

// Phi^-1(e^y) for y <= 0.
fn ndtriExp(y0: f32) -> f32 {
  let y = min(y0, -1e-30);
  var x: f32;
  if (y < -3.7191571) {
    // lower tail: q comes straight from y, no exp to cancel
    x = acklamTail(sqrt(-2.0 * y));
  } else if (y > -0.024545) {
    // upper tail: 1 - p = -y to first order, accurate exactly where exp cancels
    x = -acklamTail(sqrt(-2.0 * log(max(-y, 1e-38))));
  } else {
    let p = exp(y);
    let q = p - 0.5;
    let r = q * q;
    x = (((((-3.969683028665376e1 * r + 2.209460984245205e2) * r - 2.759285104469687e2) * r + 1.38357751867269e2) * r - 3.066479806614716e1) * r + 2.506628277459239) * q
      / (((((-5.447609879822406e1 * r + 1.615858368580409e2) * r - 1.556989798598866e2) * r + 6.680131188771972e1) * r - 1.328068155288572e1) * r + 1.0);
  }
  x = clamp(x, -37.0, 12.0);
  // Newton on log Phi; skipped above x = 5 where f32 log Phi saturates to 0
  // and the step would corrupt a good seed.
  for (var it = 0; it < 2; it++) {
    if (x >= 5.0) { break; }
    let ln = logNdtr(x);
    let step = clamp((ln - y) * exp(ln - (-0.5 * x * x - HALF_LOG_2PI)), -2.0, 2.0);
    x = x - step;
  }
  return x;
}

// ---- ping-pong ring access ------------------------------------------------

fn ringGet(readA: bool, idx: u32) -> f32 {
  if (readA) { return ringA[idx]; }
  return ringB[idx];
}

fn ringSet(readA: bool, idx: u32, v: f32) {
  if (readA) { ringB[idx] = v; } else { ringA[idx] = v; }
}

// ---- the filter -----------------------------------------------------------

var<workgroup> red: array<f32, 256>;
var<workgroup> red2: array<f32, 256>;
var<workgroup> sh: array<f32, 3>; // peak, wSum, u0

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let rep = wid.x;
  let tid = lid.x;
  let N = P.nParticles;
  let RL = P.ringLen;
  let n = P.nSteps;
  let seed = repSeeds[rep];
  // Blocked partition: each thread owns a contiguous run of particles, so
  // its local prefix sum is a contiguous slice of the cumulative weights.
  let per = (N + WG - 1u) / WG;
  let start = min(tid * per, N);
  let end = min(start + per, N);
  let pBase = rep * N;
  let rBase = rep * N * RL;

  if (P.chunkStart == 0u) {
    // The exact prior: i.i.d. standard normal innovations, no burn-in.
    for (var i = start; i < end; i++) {
      for (var j = 0u; j < RL; j++) {
        ringA[rBase + i * RL + j] = normalDraw(seed, 0xa5a50000u + j, i);
      }
    }
  }
  workgroupBarrier();
  storageBarrier();

  for (var tt = 0u; tt < P.chunkLen; tt++) {
    let t = P.chunkStart + tt;
    let readA = (t & 1u) == 0u;
    let lower = (f32(ys[rep * n + t]) - 0.5) * P.delta;

    // Weights: the Gaussian cell probability per particle, in log space. The
    // weight and the optimal proposal share their transcendentals via scratch.
    var localPeak = NEG_BIG;
    for (var i = start; i < end; i++) {
      var mu = 0.0;
      let row = rBase + i * RL;
      for (var j = 0u; j < RL; j++) {
        mu += ringGet(readA, row + j) * taps[1u + j]; // ring[j] holds W_{t-1-j}
      }
      let l = (lower - mu) / P.h0;
      let h = l + P.width;
      let flip = (l + h) > 0.0;
      var a = l;
      var b = h;
      if (flip) { a = -h; b = -l; }
      let la = logNdtr(a);
      let lb = logNdtr(b);
      let lalpha = max(lb + log1mexp(la - lb), NEG_BIG);
      scratch[pBase + i] = vec4f(la, lalpha, l, select(0.0, 1.0, flip));
      localPeak = max(localPeak, lalpha);
    }
    red[tid] = localPeak;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
      if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
      workgroupBarrier();
    }
    if (tid == 0u) { sh[0] = red[0]; }
    workgroupBarrier();
    let peak = sh[0];

    // Normalized weights and their cumulative sum: a sequential pass over
    // each thread's slice, then a Hillis-Steele scan of the 256 slice totals.
    var running = 0.0;
    var sumSq = 0.0;
    for (var i = start; i < end; i++) {
      let w = exp(scratch[pBase + i].y - peak);
      running += w;
      sumSq += w * w;
      cum[pBase + i] = running;
    }
    red[tid] = running;
    red2[tid] = sumSq;
    workgroupBarrier();
    for (var s = 1u; s < WG; s <<= 1u) {
      var carry = 0.0;
      if (tid >= s) { carry = red[tid - s]; }
      workgroupBarrier();
      red[tid] = red[tid] + carry;
      workgroupBarrier();
    }
    let wSum = red[WG - 1u];
    let offset = red[tid] - running;
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
      if (tid < s) { red2[tid] = red2[tid] + red2[tid + s]; }
      workgroupBarrier();
    }
    if (tid == 0u) {
      sh[1] = wSum;
      sh[2] = uni(seed, 0x5eed0000u, t); // u0 for systematic resampling
      let ess = wSum * wSum / max(red2[0], 1e-30) / f32(N);
      stepOut[rep * n + t] = vec2f(peak + log(wSum) - P.logN, ess);
    }
    for (var i = start; i < end; i++) {
      cum[pBase + i] += offset;
    }
    workgroupBarrier();
    storageBarrier();
    let wSumAll = sh[1];
    let u0 = sh[2];

    // Systematic resample with the exact predictive weights (binary search
    // over the cumulative sum), then the optimal proposal: the truncated
    // innovation drawn by inverse CDF from the ancestor's cell stats. The
    // gather shifts the window as it copies, so slot 0 is always W_t.
    for (var i = start; i < end; i++) {
      let u = (u0 + f32(i)) / f32(N) * wSumAll;
      var loIdx = 0u;
      var hiIdx = N;
      while (loIdx < hiIdx) {
        let mid = (loIdx + hiIdx) / 2u;
        if (cum[pBase + mid] < u) { loIdx = mid + 1u; } else { hiIdx = mid; }
      }
      let anc = min(loIdx, N - 1u);
      let s4 = scratch[pBase + anc];
      let u2 = max(uni(seed, 0xd4a30000u + t, i), 1e-12);
      let logP = logaddexp(s4.x, log(u2) + s4.y);
      var x = ndtriExp(logP);
      if (s4.w > 0.5) { x = -x; }
      let l = s4.z;
      x = clamp(x, l, l + P.width);
      let dstRow = rBase + i * RL;
      let srcRow = rBase + anc * RL;
      ringSet(readA, dstRow, x);
      for (var j = 1u; j < RL; j++) {
        ringSet(readA, dstRow + j, ringGet(readA, srcRow + j - 1u));
      }
    }
    workgroupBarrier();
    storageBarrier();
  }
}
