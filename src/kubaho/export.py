"""ONNX export, INT8 quantisation and CPU latency benchmark."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, path: str | Path, seq_len: int = 100, n_sub: int = 52) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model, torch.randn(1, seq_len, n_sub), str(path),
        input_names=["csi"], output_names=["logits"],
        dynamic_axes={"csi": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    return path


def quantize_int8(src: str | Path, dst: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return Path(dst)


def onnx_predict_proba(path: str | Path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"csi": X.astype(np.float32)})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def benchmark_ms(path: str | Path, n: int = 200, seq_len: int = 100, n_sub: int = 52) -> float:
    """Median single-window inference time on this CPU (not the ESP32-S3)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, seq_len, n_sub).astype(np.float32)
    for _ in range(20):
        sess.run(None, {"csi": x})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"csi": x})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))
