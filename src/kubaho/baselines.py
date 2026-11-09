"""M0: re-implementation of a RuView-style threshold fall detector.

A fall is declared when the motion-energy *acceleration* spikes above a threshold and
the room is comparatively still afterwards. The threshold is calibrated on generic
(source-domain) data only, exactly as an out-of-the-box detector would be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import motion_energy


def threshold_scores(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (peak acceleration, post-peak stillness ratio) for windows (N, T, 52)."""
    e = motion_energy(X)                       # (N, T-1)
    accel = np.abs(np.diff(e, axis=1))         # (N, T-2)
    peak_idx = accel.argmax(axis=1)
    peak = accel.max(axis=1)
    still = np.array([
        e[i, p + 8 :].mean() / (e[i].max() + 1e-9) if p + 8 < e.shape[1] else 1.0
        for i, p in enumerate(peak_idx)
    ])
    return peak, still


@dataclass
class ThresholdDetector:
    accel_threshold: float = 0.0
    stillness_ratio: float = 0.5

    def fit(self, X: np.ndarray, is_fall: np.ndarray, target_sensitivity: float = 0.95) -> "ThresholdDetector":
        peak, still = threshold_scores(X)
        candidates = peak[is_fall & (still < self.stillness_ratio)]
        # Highest threshold that still catches the target share of falls.
        self.accel_threshold = float(np.quantile(candidates, 1 - target_sensitivity)) if len(candidates) else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        peak, still = threshold_scores(X)
        return (peak >= self.accel_threshold) & (still < self.stillness_ratio)
