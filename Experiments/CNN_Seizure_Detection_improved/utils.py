import numpy as np
import polars as pl
from scipy import signal
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from numba import njit


def psd_prob(x, fs:float, fmin:float=0.0, fmax:float=100.0, drop_zeros:bool = False, nperseg=None):
    """Compute normalized power spectra density (PSD) (a probability distribution over frequency)."""
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    # restrict to the band the paper uses (0-100 Hz)
    mask = (f >= fmin) & (f <= fmax)
    Pxx = Pxx[mask]
    if drop_zeros:
        Pxx = Pxx[Pxx > 0]                # drop zeros so logs are safe
    p = Pxx / Pxx.sum()               # normalize -> sum(p) = 1
    return p


def spectral_entropy(x, fs:float, **kw):
    """Shannon spectral entropy, Eq. (4): SEN = sum p * log(1/p)."""
    p = psd_prob(x, fs, **kw)
    H = -np.sum(p * np.log(p))        # natural log; use np.log2 for bits
    return H

def embed(x, m:int, tau=1):
    """Delay-embed a 1-D series into m dimensions with delay tau."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    n_vectors = N - (m - 1) * tau
    if n_vectors <= 0:
        raise ValueError("Series too short for this m/tau.")
    # rows are the delay vectors X_i = [x_i, x_{i+tau}, ..., x_{i+(m-1)tau}]
    return np.array([x[i:i + (m - 1) * tau + 1:tau] for i in range(n_vectors)])


def correlation_integral(X, r, theiler=0):
    """Fraction of point pairs with Chebyshev distance < r.
    theiler w excludes pairs with |i-j| <= w (0 = paper's behavior)."""
    N = len(X)
    D = squareform(pdist(X, metric='chebyshev'))  # NxN distance matrix
    # build a mask of valid pairs (i < j, and outside the Theiler window)
    iu = np.triu_indices(N, k=theiler + 1)
    dists = D[iu]
    count = np.sum(dists < r)
    total = len(dists)
    return count / total if total > 0 else 0.0


def k2_entropy(x, m:int=10, tau=1, r=None, theiler=0):
    """Grassberger-Procaccia K2 entropy estimate (lower bound on KS entropy)."""
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.2 * np.std(x)   # a common default; see note below

    Xm  = embed(x, m,     tau)
    Xm1 = embed(x, m + 1, tau)
    # use the same number of vectors for both dimensions (fair comparison)
    n = min(len(Xm), len(Xm1))
    Xm, Xm1 = Xm[:n], Xm1[:n]

    Cm  = correlation_integral(Xm,  r, theiler)
    Cm1 = correlation_integral(Xm1, r, theiler)

    if Cm <= 0 or Cm1 <= 0:
        return np.nan  # r too small / not enough pairs at this scale
    return (1.0 / tau) * np.log(Cm / Cm1)

def _phi(x, m, r, tau=1):
    """Phi^m(r): average over points of log(fraction of points within r),
    Chebyshev distance, self-matches included (Pincus convention)."""
    X = embed(x, m, tau)
    N = len(X)
    # full NxN Chebyshev distance matrix (includes the diagonal = self-matches)
    D = squareform(pdist(X, metric='chebyshev'))
    np.fill_diagonal(D, 0.0)                 # ensure self-distance is 0 (<= r)
    C = np.sum(D <= r, axis=1) / N           # per-point fraction within r
    return np.mean(np.log(C))                # C is never 0 because self-match counts

def APEN(x, m=2, r=None, tau=1):
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.15 * np.std(x, ddof=0)         # paper: 0.15 * SD of the series
    return _phi(x, m, r, tau) - _phi(x, m + 1, r, tau)
