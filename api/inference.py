"""Loads the INT8 ONNX model (RuView encoder + Kubaho LoRA adapter, merged) for /predict."""

from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np

from kubaho.preprocess import preprocess_window

from .config import settings


class ModelNotReady(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load():
    if not settings.model_path.exists() or not settings.model_card_path.exists():
        raise ModelNotReady(
            f"Model not found at {settings.model_path}. Run notebooks/kubaho_model.ipynb (or scripts/train.py) first."
        )
    import onnxruntime as ort

    session = ort.InferenceSession(str(settings.model_path), providers=["CPUExecutionProvider"])
    card = json.loads(settings.model_card_path.read_text())
    return session, card


def model_card() -> dict:
    return _load()[1]


def predict(amplitude: np.ndarray, fs: int = 100) -> dict:
    session, card = _load()
    start = time.perf_counter()
    window = preprocess_window(amplitude, fs=fs)[None]
    logits = session.run(None, {"csi": window})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    classes = card["classes"]
    fall_p = float(p[classes.index("fall")])
    return {
        "predicted_class": classes[int(p.argmax())],
        "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)},
        "fall_probability": round(fall_p, 4),
        "fall_alert": fall_p >= card["threshold"],
        "threshold": card["threshold"],
        "model_version": card["version"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
