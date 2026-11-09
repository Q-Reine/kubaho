"""Dataset builders: Kubaho-Lab (target domain) and a generic-room source domain.

Each 3 s clip is simulated at 100 packets/s, then three raw 2 s windows (offsets
0, 0.5 and 1.0 s) are cleaned with the same `preprocess_window` the API uses, so
training and serving see identical preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES
from .preprocess import FS, packet_loss, preprocess_window
from .synthetic import (KUBAHO_ACTIVITIES, SOURCE_ACTIVITIES, SOURCE_CLASSES, make_layout,
                        make_subject, simulate_clip)

CROP_OFFSETS_S = (0.0, 0.5, 1.0)
WINDOW = int(2.0 * FS)


def _windows(csi: np.ndarray) -> np.ndarray:
    return np.stack([preprocess_window(csi[int(o * FS) : int(o * FS) + WINDOW]) for o in CROP_OFFSETS_S])


def _session_loss(rng: np.random.Generator) -> float:
    # Most sessions lose a few packets; about 1 in 12 has a bad link (>10% loss).
    return rng.uniform(0.11, 0.25) if rng.random() < 0.08 else rng.uniform(0.005, 0.05)


def build_kubaho_lab(n_subjects: int = 20, reps: int = 5, seed: int = 7) -> tuple[np.ndarray, pd.DataFrame]:
    """Mock rural room, two layouts (A, B); every volunteer records every activity in both."""
    rng = np.random.default_rng(seed)
    layouts = {name: make_layout(name, seed + i, rural=True) for i, name in enumerate("AB", start=1)}
    X, rows = [], []
    for subj in range(1, n_subjects + 1):
        subject = make_subject(rng)
        for lname, layout in layouts.items():
            loss = _session_loss(rng)
            for activity, label in KUBAHO_ACTIVITIES.items():
                for rep in range(reps):
                    csi = simulate_clip(activity, layout, subject, rng, loss_rate=loss)
                    X.append(_windows(csi))
                    rows.append(dict(subject=subj, layout=lname, session=f"S{subj:02d}-{lname}",
                                     activity=activity, label=CLASSES.index(label), class_name=label,
                                     rep=rep, packet_loss=packet_loss(csi)))
    return np.stack(X), pd.DataFrame(rows)


def build_source_domain(n_subjects: int = 36, reps: int = 8, n_layouts: int = 12,
                        seed: int = 101) -> tuple[np.ndarray, pd.DataFrame]:
    """Many generic office/lab rooms with chairs and hard floors, and no floor-level activities.

    A real pretrained encoder (RuView, public datasets) has seen many rooms; using 12 layouts keeps the
    stand-in comparable instead of over-fitting it to a handful of rooms.
    """
    rng = np.random.default_rng(seed)
    layouts = [make_layout(f"office{i}", seed + i, rural=False) for i in range(n_layouts)]
    X, rows = [], []
    for subj in range(n_subjects):
        subject, layout = make_subject(rng), layouts[subj % n_layouts]
        for activity, label in SOURCE_ACTIVITIES.items():
            for rep in range(reps):
                csi = simulate_clip(activity, layout, subject, rng, loss_rate=rng.uniform(0.005, 0.03))
                X.append(_windows(csi))
                rows.append(dict(subject=subj, layout=layout.name, activity=activity,
                                 label=SOURCE_CLASSES.index(label), class_name=label, rep=rep))
    return np.stack(X), pd.DataFrame(rows)


def load_or_build(name: str, cache_dir: str | Path = "data/processed", **kwargs) -> tuple[np.ndarray, pd.DataFrame]:
    builder = {"kubaho_lab": build_kubaho_lab, "source": build_source_domain}[name]
    cache_dir = Path(cache_dir)
    npz, csv = cache_dir / f"{name}.npz", cache_dir / f"{name}_meta.csv"
    if npz.exists() and csv.exists():
        return np.load(npz)["X"], pd.read_csv(csv)
    X, meta = builder(**kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, X=X)
    meta.to_csv(csv, index=False)
    return X, meta
