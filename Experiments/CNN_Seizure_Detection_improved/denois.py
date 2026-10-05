import numpy as np
from math import log, sqrt
from statistics import median
import matplotlib.pyplot as plt
import pywt


def plot(signal:np.array[float], title:str = "") -> None:
    # signal: 1-D EEG, fs: sampling rate in Hz
    fs = 256
    dt = 1 / fs

    # choose scales -> these map to frequencies
    # use a continuous wavelet: 'morl' (Morlet), 'cmor' (complex Morlet), 'mexh'
    wavelet = "shan"#'cmor1.5-1.0'   # complex Morlet: bandwidth 1.5, center freq 1.0

    # pick the frequencies you care about (e.g. 1-40 Hz for EEG) and convert to scales
    freqs = np.arange(1, 100)                          # 1..40 Hz
    scales = pywt.frequency2scale(wavelet, freqs / fs)  # note: normalized freq

    coefs, freqs_out = pywt.cwt(signal, scales, wavelet, sampling_period=dt)

    # coefs is complex for cmor -> power is |coefs|^2
    power = np.abs(coefs) ** 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 4), dpi=100, sharex=True,
                                    gridspec_kw={'height_ratios': [1, 2]})

    t = np.arange(len(signal)) * dt

    # Top: signal
    ax1.plot(t, signal)
    ax1.set_ylabel('Amplitude')

    if title != "":
        ax1.set_title(title)

    # Bottom: spectrogram
    pcm = ax2.pcolormesh(t, freqs_out, power, shading='gouraud')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    fig.colorbar(pcm, ax=[ax1, ax2], label='Power')
    plt.show()


def global_threshold(signal:np.array[float], coefs:np.array[float]) -> float:
    noise_std = median(np.abs(coefs))/0.6745

    return noise_std * sqrt(2*log(np.size(signal)))


def global_threshold_denoising(signal, wavelet="sym4"):
    cA, cD = pywt.dwt(signal, wavelet, mode="periodic")
    threshold = global_threshold(signal, cD)   # sigma from cD
    cD = np.where(np.abs(cD) > threshold, 0, cD)   # remove ARTIFACTS (large)
    return pywt.idwt(cA, cD, wavelet, mode="periodic")


def STD_Threshold(coefs):
    return 1.5 * np.std(np.concatenate(coefs[1:]))   # exclude cA


def STD_threshold_denoising(signal, wavelet="sym4", level=1):
    coef = pywt.wavedec(signal, wavelet, mode="periodic", level=level)
    threshold = STD_Threshold(coef)
    for i in range(1, len(coef)):
        coef[i] = np.where(np.abs(coef[i]) > threshold, 0, coef[i])  # remove large
    return pywt.waverec(coef, wavelet, mode="periodic")


def hight_pass_filter(coeffs:nparray[float], f:float) -> np.array[float]:
    return np.where(coeffs > f, coeffs, 0)

import numpy as np
import pywt


def _theta_alpha(coeffs, beta, k1, k2):
    """theta_alpha from IQR of coefficients (Eq. 22)."""
    q75, q25 = np.percentile(coeffs, [75, 25])
    r = q75 - q25                      # r = IQR(w)
    f_beta_r = k2 * np.exp(-beta / 100.0 * k2 * r * r)
    return f_beta_r if f_beta_r >= k1 else k1


def _lambda_elimination(w, t_alpha):
    """Eq. 19: keep if |w| <= theta_alpha, else 0."""
    out = w.copy()
    out[np.abs(w) > t_alpha] = 0.0
    return out


def _lambda_linear(w, t_alpha, t_beta):
    """Eq. 20: linear attenuation."""
    out = w.copy()
    a = np.abs(w)
    mid = (a > t_alpha) & (a <= t_beta)
    out[mid] = np.sign(w[mid]) * t_alpha * (
        1.0 - (a[mid] - t_alpha) / (t_beta - t_alpha)
    )
    out[a > t_beta] = 0.0
    return out


def _lambda_soft(w, t_alpha, t_gamma):
    """Eq. 21: soft thresholding."""
    # alpha = -(1/theta_gamma) * log( (theta_alpha - theta_gamma)/(theta_alpha + theta_gamma) )
    alpha = -(1.0 / t_gamma) * np.log(
        (t_alpha - t_gamma) / (t_alpha + t_gamma)
    )
    out = w.copy()
    a = np.abs(w)
    hi = a >= t_gamma
    ww = w[hi]
    out[hi] = (1.0 - np.exp(-alpha * ww)) / (1.0 + np.exp(-alpha * ww)) * t_alpha
    return out


def _filter_window(w, mode, beta, k1, k2, theta_gamma_ratio, theta_beta_ratio):
    t_alpha = _theta_alpha(w, beta, k1, k2)
    t_gamma = theta_gamma_ratio * t_alpha     # default 0.8 * theta_alpha
    t_beta = theta_beta_ratio * t_alpha       # default 2.0 * theta_alpha

    if mode == "elim":
        return _lambda_elimination(w, t_alpha)
    elif mode == "linear":
        return _lambda_linear(w, t_alpha, t_beta)
    elif mode == "soft":
        return _lambda_soft(w, t_alpha, t_gamma)
    else:
        raise ValueError("mode must be 'elim', 'linear', or 'soft'")


def atar(x, fs=256, wavelet="db3", level=None, mode="soft",
         beta=0.1, k1=10.0, k2=100.0,
         theta_gamma_ratio=0.8, theta_beta_ratio=2.0,
         win_sec=1.0, overlap=0.5):
    """
    ATAR: Automatic and Tunable Artefact Removal (Bajaj et al., 2020).

    Pipeline (Fig. 8): window -> L-level WPD -> filter coeffs (Eq. 19/20/21)
    with theta_alpha from IQR (Eq. 22) -> IWPD -> overlap-add.

    Parameters
    ----------
    x        : 1-D EEG signal
    mode     : 'elim' | 'linear' | 'soft'
    beta     : tuning parameter (higher -> more aggressive removal)
    k1, k2   : lower / upper bounds on the threshold
    win_sec  : window length in seconds
    overlap  : fractional overlap between windows (0..1)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    win = int(round(win_sec * fs))
    if level is None:
        level = pywt.dwt_max_level(win, pywt.Wavelet(wavelet).dec_len)
    step = max(1, int(round(win * (1.0 - overlap))))

    y = np.zeros(n)
    wsum = np.zeros(n)
    window_fn = np.hanning(win)

    starts = list(range(0, max(1, n - win + 1), step))
    if starts[-1] != n - win and n >= win:
        starts.append(n - win)

    for s in starts:
        seg = x[s:s + win]
        if len(seg) < win:                      # pad last short segment
            seg = np.pad(seg, (0, win - len(seg)))

        wp = pywt.WaveletPacket(seg, wavelet, mode="periodization", maxlevel=level)
        nodes = wp.get_level(level, order="natural")

        for node in nodes:
            node.data = _filter_window(
                np.asarray(node.data), mode, beta, k1, k2,
                theta_gamma_ratio, theta_beta_ratio,
            )

        rec = wp.reconstruct(update=True)[:win]

        y[s:s + len(rec)] += rec * window_fn[:len(rec)]
        wsum[s:s + len(rec)] += window_fn[:len(rec)]

    wsum[wsum == 0] = 1.0
    return y / wsum