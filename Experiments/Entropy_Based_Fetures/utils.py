import numpy as np
import polars as pl
from scipy import signal
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from numba import njit



def plot_entropy_distribution(df, value_col="spectral_entropy", class_col="classe",
                              bins=30, xlabel="Spectral entropy"):
    """Plot per-class histograms of an entropy feature as separate panels.

    df is a Polars DataFrame with a numeric `value_col` and a categorical `class_col`.
    """

    classes = df[class_col].unique().to_list()
    fig, axes = plt.subplots(1, len(classes),
                             figsize=(5 * len(classes), 4), sharex=True)

    # ensure axes is iterable when there is only one class
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        vals = df.filter(pl.col(class_col) == cls)[value_col].to_numpy()
        ax.hist(vals, bins=bins)
        ax.set_title(str(cls))
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Count")

    plt.tight_layout()
    plt.show()
    return fig

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

def renyi_entropy(x, fs:float, alpha:float=2.0, **kw):
    """Renyi entropy, Eq. (5). alpha=2 -> Eq. (6), the quadratic case."""
    p = psd_prob(x, fs, **kw)
    if np.isclose(alpha, 1.0):        # limit case = Shannon
        return -np.sum(p * np.log(p))
    return (1.0 / (1.0 - alpha)) * np.log(np.sum(p ** alpha))

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

def k2_entropy(x, m=10, tau=1, r=None, theiler=0):
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