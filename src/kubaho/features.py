"""Hand-crafted statistical features for the classical baselines (M1-M3)."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from .preprocess import TARGET_FS, motion_energy

BANDS_HZ = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 25)]

FEATURE_NAMES = (
    ["mean_std", "mean_range", "mean_skew", "mean_kurt",
     "energy_mean", "energy_max", "energy_std", "energy_peak_pos", "energy_peak_ratio",
     "energy_pre_peak", "energy_post_peak", "energy_last_quarter", "post_peak_stillness",
     "accel_max", "accel_mean", "total_var", "pc1_var_ratio", "pc1_dom_freq"]
    + [f"pc1_band_{lo}_{hi}hz" for lo, hi in BANDS_HZ]
)


def window_features(w: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """w: cleaned amplitude window (T, 52) -> feature vector."""
    m = w.mean(axis=1)
    e = motion_energy(w)
    peak = int(e.argmax())
    pre = e[:peak].mean() if peak > 0 else e[0]
    post = e[peak + 5 :].mean() if peak + 5 < len(e) else e[-1]
    accel = np.abs(np.diff(e))

    centred = w - w.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    pc1 = centred @ vt[0]
    spec = np.abs(np.fft.rfft(pc1)) ** 2
    freqs = np.fft.rfftfreq(len(pc1), 1 / fs)
    total = spec.sum() + 1e-12
    bands = [spec[(freqs >= lo) & (freqs < hi)].sum() / total for lo, hi in BANDS_HZ]

    return np.array(
        [m.std(), np.ptp(m), skew(m), kurtosis(m),
         e.mean(), e.max(), e.std(), peak / len(e), e.max() / (e.mean() + 1e-9),
         pre, post, e[-len(e) // 4 :].mean(), post / (e.max() + 1e-9),
         accel.max(), accel.mean(), centred.var(), s[0] ** 2 / ((s ** 2).sum() + 1e-12),
         freqs[spec[1:].argmax() + 1]]
        + bands,
        dtype=np.float32,
    )


def extract_features(X: np.ndarray) -> np.ndarray:
    """X: (N, T, 52) -> (N, n_features)."""
    return np.nan_to_num(np.stack([window_features(w) for w in X]))
