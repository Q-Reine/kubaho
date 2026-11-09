"""Physics-inspired synthetic WiFi-CSI generator.

This module stands in for the Kubaho-Lab recordings (20 volunteers in a mock rural
room) until the real ESP32-S3 sessions are collected. It is NOT a substitute for the
real data: every result produced from it must be reported as preliminary.

Signal model (per packet t, subcarrier k):

    H[t, k] = H_static[k]                                   # room multipath
            + g(h_t) * ( B[k] e^{-j 2pi 2 d_t / lambda_k}     # direct body reflection
                       + rho * F[k] e^{-j 2pi (2 d_t + 2 h_t) / lambda_k} )  # floor bounce
            + impact burst + noise

where d_t is the radial position of the body, h_t the height of the torso centre and
g(h) how visible the body is at that height. Layouts change H_static, B, F, rho and
the floor-level visibility, which is what creates the domain gap between generic
(office/lab) rooms and rural Rwandan rooms with mats and low furniture.

The receiver adds per-packet AGC jitter, impulsive outliers, packet loss and random
sampling-time / carrier-frequency phase offsets, which the preprocessing removes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 100                     # CSI packets per second (ESP32-S3 rate in the proposal)
CLIP_S = 3.0                 # each recorded event is a 3 s clip
SUB_IDX = np.r_[-26:0, 1:27]  # 52 HT20 subcarriers (DC excluded)
N_SUB = len(SUB_IDX)
F_C = 2.437e9                # WiFi channel 6
DF = 312.5e3                 # subcarrier spacing
LAMBDAS = 3e8 / (F_C + SUB_IDX * DF)

# Kubaho-Lab protocol: 4 fall types + 6 fall-like / daily activities + walk + empty room.
KUBAHO_ACTIVITIES = {
    "fall_forward": "fall",
    "fall_backward": "fall",
    "fall_sideways": "fall",
    "fall_from_stool": "fall",
    "sit_on_mat": "sit_lie_mat",
    "lie_on_mat": "sit_lie_mat",
    "kneel_pray": "kneel_pray",
    "squat_stool": "squat_stool",
    "bend_sweep": "bend_sweep",
    "restless_bed": "restless_bed",
    "walk": "walk",
    "empty": "empty",
}

# Generic-room data used to pre-train the encoder (stand-in for public datasets).
SOURCE_ACTIVITIES = {
    "fall_hard_floor": "fall",
    "walk": "walk",
    "sit_chair": "sit_chair",
    "stand_up": "stand_up",
    "bend_pick": "bend_pick",
    "empty": "empty",
}
SOURCE_CLASSES = ["fall", "walk", "sit_chair", "stand_up", "bend_pick", "empty"]


@dataclass
class Layout:
    name: str
    static: np.ndarray       # (N_SUB,) complex room response
    body_resp: np.ndarray    # (N_SUB,) complex response of the direct body path
    floor_resp: np.ndarray   # (N_SUB,) complex response of the floor-bounce path
    floor_rho: float         # strength of the floor-bounce path
    floor_gain: float        # how visible a person lying on the floor is
    r0: float                # distance of the activity zone from the node (m)


@dataclass
class Subject:
    h0: float     # standing torso-centre height (m)
    speed: float  # movement speed multiplier
    gain: float   # body reflectivity multiplier


def _freq_selective(rng: np.random.Generator, n_paths: int, scale: float) -> np.ndarray:
    delays = rng.uniform(5, 60, n_paths) * 1e-9
    gains = scale * rng.rayleigh(1.0, n_paths) / np.sqrt(n_paths)
    phases = rng.uniform(0, 2 * np.pi, n_paths)
    freqs = F_C + SUB_IDX * DF
    return (gains[:, None] * np.exp(1j * (phases[:, None] - 2 * np.pi * freqs[None, :] * delays[:, None]))).sum(0)


def make_layout(name: str, seed: int, rural: bool) -> Layout:
    """Rural rooms: mats and low stools, nothing blocking the floor, strong floor bounce.
    Generic rooms: tables and chairs hide the floor, weaker floor bounce."""
    rng = np.random.default_rng(seed)
    return Layout(
        name=name,
        static=_freq_selective(rng, 6, 3.0),
        body_resp=_freq_selective(rng, 2, 1.0),
        floor_resp=_freq_selective(rng, 2, 1.0),
        floor_rho=rng.uniform(0.5, 0.8) if rural else rng.uniform(0.15, 0.3),
        floor_gain=rng.uniform(0.7, 0.9) if rural else rng.uniform(0.35, 0.5),
        r0=rng.uniform(2.0, 3.5),
    )


def make_subject(rng: np.random.Generator) -> Subject:
    return Subject(h0=rng.uniform(0.92, 1.08), speed=rng.uniform(0.8, 1.25), gain=rng.uniform(0.8, 1.2))


def _smooth(u: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))


def _trajectory(activity: str, t: np.ndarray, s: Subject, rng: np.random.Generator):
    """Return torso height h(t), radial position r(t) and impact envelope for an activity."""
    T = len(t)
    h = np.full(T, s.h0)
    r = np.full(T, rng.uniform(-0.2, 0.2))
    impact = np.zeros(T)
    t0 = rng.uniform(0.9, 1.3)  # event onset, inside the centred 2 s window

    if activity.startswith("fall"):
        hard = activity == "fall_hard_floor"
        if activity == "fall_from_stool":
            h_start, dur = 0.45 * s.h0, rng.uniform(0.3, 0.45) / s.speed
        elif hard:
            h_start, dur = s.h0, rng.uniform(0.35, 0.55) / s.speed
        else:
            h_start, dur = s.h0, rng.uniform(0.45, 0.7) / s.speed
        h_end = rng.uniform(0.1, 0.18)
        frac = np.clip((t - t0) / dur, 0, 1) ** 2  # accelerating, gravity-like descent
        h = h_start - (h_start - h_end) * frac
        lateral = {"fall_forward": 0.6, "fall_backward": -0.5, "fall_sideways": 0.2,
                   "fall_from_stool": 0.3, "fall_hard_floor": 0.6}[activity]
        r = r + lateral * frac
        t1 = t0 + dur
        after = t >= t1
        h = h + after * 0.03 * np.exp(-(t - t1) / 0.08) * np.abs(np.sin(2 * np.pi * 6 * (t - t1)))
        # Lab falls land on crash mats (softer) - public/generic falls on hard floors.
        impact = np.exp(-(((t - t1) / 0.03) ** 2)) * rng.uniform(0.8, 1.4) * (1.8 if hard else 0.8)
    elif activity == "sit_on_mat":
        dur = rng.uniform(1.1, 1.8) / s.speed
        h = s.h0 - (s.h0 - 0.38) * _smooth((t - t0) / dur)
    elif activity == "lie_on_mat":
        d1, d2 = rng.uniform(0.9, 1.3) / s.speed, rng.uniform(0.6, 0.9) / s.speed
        h = s.h0 - (s.h0 - 0.38) * _smooth((t - t0) / d1) - (0.38 - 0.14) * _smooth((t - t0 - d1) / d2)
        r = r + 0.4 * _smooth((t - t0 - d1) / d2)
    elif activity == "kneel_pray":
        dur = rng.uniform(0.8, 1.3) / s.speed
        h = s.h0 - (s.h0 - 0.5) * _smooth((t - t0) / dur)
        f = rng.uniform(0.35, 0.6)
        h = h - 0.15 * (0.5 - 0.5 * np.cos(2 * np.pi * f * (t - t0 - dur))) * (t > t0 + dur)
    elif activity == "squat_stool":
        dur = rng.uniform(0.5, 0.85) / s.speed
        h = s.h0 - (s.h0 - 0.42) * _smooth((t - t0) / dur)
        impact = 0.25 * np.exp(-(((t - t0 - dur) / 0.04) ** 2))
    elif activity == "bend_sweep":
        f = rng.uniform(0.8, 1.4)
        h = 0.72 * s.h0 + 0.05 * np.sin(2 * np.pi * f * t)
        r = r + 0.25 * np.sin(2 * np.pi * f * t + rng.uniform(0, np.pi))
    elif activity == "restless_bed":
        h = np.full(T, 0.5)
        for _ in range(rng.integers(1, 4)):
            tb, dur = rng.uniform(0.3, 2.4), rng.uniform(0.4, 0.9)
            u = (t - tb) / dur
            bump = np.sin(np.pi * np.clip(u, 0, 1))
            h = h + 0.12 * bump * rng.choice([-1, 1])
            r = r + 0.2 * bump
    elif activity == "walk":
        v = rng.uniform(0.6, 1.2) * s.speed * rng.choice([-1, 1])
        r = r + v * (t - 1.5)
        h = s.h0 + 0.025 * np.sin(2 * np.pi * 1.8 * s.speed * t)
    elif activity == "sit_chair":
        dur = rng.uniform(0.7, 1.2) / s.speed
        h = s.h0 - (s.h0 - 0.62) * _smooth((t - t0) / dur)
    elif activity == "stand_up":
        dur = rng.uniform(0.8, 1.3) / s.speed
        h = 0.62 + (s.h0 - 0.62) * _smooth((t - t0) / dur)
    elif activity == "bend_pick":
        dur = rng.uniform(1.2, 1.8) / s.speed
        h = s.h0 - (s.h0 - 0.6) * np.sin(np.pi * np.clip((t - t0) / dur, 0, 1))
    elif activity != "empty":
        raise ValueError(f"unknown activity {activity!r}")

    # Breathing keeps a still person faintly visible.
    r = r + 0.003 * np.sin(2 * np.pi * rng.uniform(0.2, 0.33) * t + rng.uniform(0, 2 * np.pi))
    return h, r, impact


def simulate_clip(activity: str, layout: Layout, subject: Subject, rng: np.random.Generator,
                  fs: int = FS, duration: float = CLIP_S, loss_rate: float = 0.0) -> np.ndarray:
    """Simulate one CSI recording. Returns complex (T, N_SUB); lost packets are NaN rows."""
    t = np.arange(int(fs * duration)) / fs
    T = len(t)
    csi = np.repeat(layout.static[None, :], T, axis=0).astype(np.complex128)
    # Slow environmental drift (doors, people outside the room).
    csi *= 1 + 0.01 * np.sin(2 * np.pi * rng.uniform(0.02, 0.1) * t + rng.uniform(0, 6))[:, None]

    if activity != "empty":
        h, r, impact = _trajectory(activity, t, subject, rng)
        vis = layout.floor_gain + (1 - layout.floor_gain) * np.clip((h - 0.15) / 0.85, 0, 1)
        g = subject.gain * vis
        d = layout.r0 + r
        direct = layout.body_resp[None, :] * np.exp(-2j * np.pi * (2 * d)[:, None] / LAMBDAS[None, :])
        bounce = layout.floor_resp[None, :] * np.exp(-2j * np.pi * (2 * d + 2 * h)[:, None] / LAMBDAS[None, :])
        csi += g[:, None] * (direct + layout.floor_rho * bounce)
        if impact.any():
            burst = (rng.normal(size=(T, N_SUB)) + 1j * rng.normal(size=(T, N_SUB))) / np.sqrt(2)
            csi += 0.8 * impact[:, None] * burst

    scale = np.abs(layout.static).mean()
    csi *= np.exp(rng.normal(0, 0.02, (T, 1)))  # AGC jitter
    csi += 0.04 * scale * (rng.normal(size=(T, N_SUB)) + 1j * rng.normal(size=(T, N_SUB))) / np.sqrt(2)
    outliers = rng.random(T) < 0.005
    csi[outliers] *= rng.uniform(2, 4, (outliers.sum(), 1))
    # Sampling-time offset (linear phase slope) and carrier-frequency offset per packet.
    sto = rng.uniform(-0.6, 0.6, (T, 1))
    cfo = rng.uniform(-np.pi, np.pi, (T, 1))
    csi *= np.exp(1j * (cfo + sto * np.pi * SUB_IDX[None, :] / 26))

    if loss_rate > 0:
        csi[rng.random(T) < loss_rate] = np.nan
    return csi
