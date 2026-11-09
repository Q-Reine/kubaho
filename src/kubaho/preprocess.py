"""CSI quality phase: packet-loss repair, phase sanitisation, Hampel filter, band-pass, windows.

Only numpy/scipy are used so the API container can import this without PyTorch.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

FS = 100          # raw CSI rate (packets/s)
TARGET_FS = 50    # rate after decimation; human motion is below 20 Hz
WINDOW_S = 2.0
STRIDE_S = 0.25
MAX_LOSS = 0.10   # sessions losing more than 10% of packets are dropped


def packet_loss(csi: np.ndarray) -> float:
    return float(np.isnan(csi).any(axis=1).mean())


def interpolate_lost(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN rows (lost packets) along time, per subcarrier."""
    lost = np.isnan(x).any(axis=1)
    if not lost.any():
        return x
    x = x.copy()
    t = np.arange(len(x))
    keep = ~lost
    for part in ([x.real, x.imag] if np.iscomplexobj(x) else [x]):
        for k in range(x.shape[1]):
            part[lost, k] = np.interp(t[lost], t[keep], part[keep, k])
    return x


def sanitize_phase(csi: np.ndarray) -> np.ndarray:
    """Remove the per-packet linear phase slope (timing offset) and constant (CFO)."""
    phase = np.unwrap(np.angle(csi), axis=1)
    k = np.arange(phase.shape[1])
    slope = (phase[:, -1] - phase[:, 0]) / (k[-1] - k[0])
    detrended = phase - slope[:, None] * k[None, :]
    return detrended - detrended.mean(axis=1, keepdims=True)


def hampel(x: np.ndarray, half_window: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace samples further than n_sigmas robust deviations from the rolling median."""
    size = (2 * half_window + 1, 1)
    med = median_filter(x, size=size, mode="nearest")
    mad = 1.4826 * median_filter(np.abs(x - med), size=size, mode="nearest")
    return np.where(np.abs(x - med) > n_sigmas * mad + 1e-9, med, x)


def bandpass(x: np.ndarray, fs: int = FS, low: float = 0.2, high: float = 20.0, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def preprocess_window(csi: np.ndarray, fs: int = FS, target_fs: int = TARGET_FS) -> np.ndarray:
    """Raw window (complex CSI or amplitude, shape (fs*2, 52)) -> cleaned amplitude (target_fs*2, 52)."""
    x = interpolate_lost(np.asarray(csi))
    amp = np.abs(x) if np.iscomplexobj(x) else x.astype(np.float64)
    amp = hampel(amp)
    amp = bandpass(amp, fs=fs)
    return amp[:: fs // target_fs].astype(np.float32)


def sliding_windows(x: np.ndarray, fs: int, window_s: float = WINDOW_S, stride_s: float = STRIDE_S) -> np.ndarray:
    win, stride = int(window_s * fs), int(stride_s * fs)
    starts = range(0, len(x) - win + 1, stride)
    return np.stack([x[s : s + win] for s in starts])


def motion_energy(window: np.ndarray) -> np.ndarray:
    """Mean absolute frame-to-frame amplitude change across subcarriers."""
    return np.abs(np.diff(window, axis=-2)).mean(axis=-1)
