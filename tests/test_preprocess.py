import numpy as np

from kubaho.features import FEATURE_NAMES, window_features
from kubaho.preprocess import hampel, packet_loss, preprocess_window, sanitize_phase
from kubaho.synthetic import make_layout, make_subject, simulate_clip


def test_preprocess_window_shape_and_loss_repair():
    rng = np.random.default_rng(0)
    csi = simulate_clip("fall_forward", make_layout("A", 1, True), make_subject(rng), rng, loss_rate=0.05)
    assert 0 < packet_loss(csi) < 0.15
    w = preprocess_window(csi[:200])
    assert w.shape == (100, 52) and np.isfinite(w).all()
    assert len(window_features(w)) == len(FEATURE_NAMES)


def test_sanitize_phase_removes_linear_slope():
    k = np.arange(52)
    csi = np.exp(1j * (0.3 * k + 1.0))[None, :].repeat(4, 0)
    assert np.abs(sanitize_phase(csi)).max() < 1e-6


def test_hampel_replaces_spike():
    x = np.ones((50, 1))
    x[25] = 100
    assert hampel(x)[25, 0] == 1
