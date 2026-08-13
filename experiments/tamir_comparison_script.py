"""Compare, for the unit-step quantized Gaussian MA(1) process X_n = W_n + theta*W_{n-1}
(sigma = 1, Delta = 1, all quantities in NATS to match Tamir's Fig. 3a):

  1. Tamir (2022/2023) Theorem 1 bound:  1/2 log(2pi e) + (1/4pi) int log(Phi_Y + 1/12)
     -- uses the PSD Phi_Y of the QUANTIZED process (requires R_Y(0), R_Y(1)).
  2. Conditional entropy bound H(Y_{n+1} | Y_n)  (Tamir's H_CE).
  3. Manuscript approximation: (1/4pi) int log(2pi e (S(w) + 1/12))  with S the INPUT spectrum.
  4. "True" entropy rate via a fully adapted particle filter (manuscript Sec. 4).
"""
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.integrate import quad

rng = np.random.default_rng(1)

def input_spectrum(theta):
    return lambda w: 1 + theta**2 + 2*theta*np.cos(w)

def manuscript_approx(theta):
    """(1/4pi) int_{-pi}^{pi} log(2 pi e (S(w) + 1/12)) dw   [nats, Delta=1]"""
    S = input_spectrum(theta)
    f = lambda w: np.log(2*np.pi*np.e*(S(w) + 1/12.0))
    val, _ = quad(f, 0, np.pi, limit=400)
    return val/np.pi/2 * 2 / 2 * 2  # (1/(4pi)) * 2*int_0^pi = int_0^pi f / (2pi)
    
def manuscript_approx2(theta):
    S = input_spectrum(theta)
    f = lambda w: np.log(2*np.pi*np.e*(S(w) + 1/12.0))
    val, _ = quad(f, 0, np.pi, limit=400)
    return val/(2*np.pi)*2/2 + 0  # placeholder

def approx_formula(theta):
    S = input_spectrum(theta)
    f = lambda w: np.log(2*np.pi*np.e*(S(w) + 1/12.0))
    val, _ = quad(f, 0, np.pi, limit=400)   # int over [0,pi]; full integral = 2*val
    return 2*val/(4*np.pi)

def classical_formula(theta):
    S = input_spectrum(theta)
    f = lambda w: np.log(np.maximum(2*np.pi*np.e*S(w), 1e-300))
    val, _ = quad(f, 0, np.pi, limit=800)
    return 2*val/(4*np.pi)

# ---------- Tamir Theorem-1 bound: needs R_Y(0)=F and R_Y(1)=G ----------
KMAX = 40

def EQ2(sd):
    """E[Q(Z)^2] for Z ~ N(0, sd^2), unit-step mid-tread quantizer."""
    k = np.arange(1, KMAX+1)
    p = ndtr((k+0.5)/sd) - ndtr((k-0.5)/sd)
    return float(np.sum(2*k**2*p))

def EQ_mean(mu, sd):
    """E[Q(Z)] for Z ~ N(mu, sd^2); mu may be an array."""
    m = np.arange(-KMAX, KMAX+1)[:, None]
    mu = np.atleast_1d(mu)[None, :]
    p = ndtr((m + 0.5 - mu)/sd) - ndtr((m - 0.5 - mu)/sd)
    return np.sum(m*p, axis=0)

def tamir_FG(theta):
    F = EQ2(np.sqrt(1+theta**2))
    # G = int E[Q(W+theta s)] E[Q(s+theta W')] phi(s) ds ; W,W' ~ N(0,1)
    s = np.linspace(-8, 8, 4001)
    phi = np.exp(-s**2/2)/np.sqrt(2*np.pi)
    A = EQ_mean(theta*s, 1.0)          # E[Q(W_{n+1}+theta s)]
    B = EQ_mean(s, max(theta, 1e-12))  # E[Q(s+theta W_{n-1})]
    G = np.trapezoid(A*B*phi, s)
    return F, G

def tamir_th1(theta):
    F, G = tamir_FG(theta)
    f = lambda lam: np.log(F + 1/12.0 + 2*G*np.cos(lam))
    val, _ = quad(f, 0, np.pi, limit=400)
    return 0.5*np.log(2*np.pi*np.e) + 2*val/(4*np.pi), F, G

# ---------- conditional entropy bound H(Y1|Y0) ----------
def cond_entropy(theta):
    K = 30
    s = np.linspace(-8, 8, 3001)
    phi = np.exp(-s**2/2)/np.sqrt(2*np.pi)
    i = np.arange(-K, K+1)
    # J0(i,s): P(i-.5 <= s + theta W <= i+.5), W~N(0,1) -> std theta
    sdJ = max(theta, 1e-12)
    J = ndtr((i[:, None] + 0.5 - s[None, :])/sdJ) - ndtr((i[:, None] - 0.5 - s[None, :])/sdJ)
    # I0(j,s): P(j-.5 <= W + theta s <= j+.5) -> std 1, mean theta s
    I = ndtr((i[:, None] + 0.5 - theta*s[None, :])/1.0) - ndtr((i[:, None] - 0.5 - theta*s[None, :])/1.0)
    # joint P(i,j) = int J(i,s) I(j,s) phi(s) ds
    W = (J*phi[None, :])
    P = np.trapezoid(W[:, None, :]*I[None, :, :], s, axis=2)  # (i,j)
    P = np.clip(P, 0, None); P /= P.sum()
    Pi = P.sum(axis=1)
    Hjoint = -np.sum(P[P > 0]*np.log(P[P > 0]))
    Hmarg = -np.sum(Pi[Pi > 0]*np.log(Pi[Pi > 0]))
    return Hjoint - Hmarg, Hmarg

# ---------- fully adapted particle filter (manuscript Sec. 4) ----------
def pf_rate(theta, n=100_000, N=2000, reps=4, seed=0):
    # minimum-phase taps
    if abs(theta) <= 1: h0, h1 = 1.0, theta
    else:               h0, h1 = abs(theta), np.sign(theta)*1.0
    ests = []
    for r in range(reps):
        g = np.random.default_rng(seed + 1000*r)
        W = g.standard_normal(n+1)
        X = h0*W[1:] + h1*W[:-1]
        Y = np.rint(X).astype(np.int64)
        wprev = g.standard_normal(N)   # exact prior
        ll = 0.0
        for t in range(n):
            mu = h1*wprev
            l = (Y[t] - 0.5 - mu)/h0
            rr = l + 1.0/h0
            Fl, Fr = ndtr(l), ndtr(rr)
            a = np.clip(Fr - Fl, 1e-300, None)
            ll += np.log(a.mean())
            # systematic resampling
            c = np.cumsum(a); c /= c[-1]
            u = (g.random() + np.arange(N))/N
            idx = np.searchsorted(c, u)
            l, Fl, Fr = l[idx], Fl[idx], Fr[idx]
            # exact truncated-normal draw
            u2 = g.random(N)
            wprev = ndtri(Fl + u2*(Fr - Fl))
            wprev = np.clip(wprev, l, l + 1.0/h0)
        ests.append(-ll/n)
    ests = np.array(ests)
    return ests.mean(), ests.std(ddof=1)/np.sqrt(reps)

# ---------- exact marginal entropy (theta = 0 sanity check) ----------
def exact_marginal(sd):
    k = np.arange(-200, 201)
    p = ndtr((k+0.5)/sd) - ndtr((k-0.5)/sd)
    p = p[p > 0]
    return -np.sum(p*np.log(p))

print(f"theta=0 sanity: exact H(Y) = {exact_marginal(1.0):.6f} nats; "
      f"manuscript approx = {approx_formula(0.0):.6f} nats; "
      f"Tamir TH-1 = {tamir_th1(0.0)[0]:.6f} nats")
print()
hdr = f"{'theta':>5} | {'classical':>9} | {'manuscript':>10} | {'PF (true)':>18} | {'CE bound':>8} | {'Tamir TH-1':>10}"
print(hdr); print("-"*len(hdr))
for theta in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    cl = classical_formula(theta)
    ap = approx_formula(theta)
    th1, F, G = tamir_th1(theta)
    ce, _ = cond_entropy(theta)
    pf, se = pf_rate(theta)
    print(f"{theta:5.2f} | {cl:9.4f} | {ap:10.4f} | {pf:9.4f} +/- {se:.4f} | {ce:8.4f} | {th1:10.4f}")
