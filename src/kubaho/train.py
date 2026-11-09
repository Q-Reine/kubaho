"""Training loop: class-weighted focal loss, AdamW, cosine schedule, early stopping,
random-crop and per-subcarrier gain augmentation."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

CENTER_CROP = 1  # crops are taken at 0 s, 0.5 s and 1.0 s of each 3 s clip


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 5
    focal_gamma: float = 2.0
    seed: int = 0


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = counts.sum() / np.maximum(counts, 1) / n_classes
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def focal_loss(logits: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor, gamma: float) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
    return (-alpha[y] * (1 - logp.exp()) ** gamma * logp).mean()


def augment_gain(X: np.ndarray, rng: np.random.Generator, strength: float = 0.3) -> np.ndarray:
    """Random smooth per-subcarrier gain, simulating how a different room reshapes the channel."""
    n, _, s = X.shape
    k = np.arange(s) / s
    freq = rng.uniform(0.5, 3.0, (n, 1, 1))
    phase = rng.uniform(0, 2 * np.pi, (n, 1, 1))
    log_gain = rng.normal(0, strength, (n, 1, 1)) + strength * np.sin(2 * np.pi * freq * k + phase)
    return (X * np.exp(log_gain)).astype(np.float32)


def _crop(X: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    """X is (N, crops, T, S): random crop per sample when training, centre crop otherwise."""
    if X.ndim == 3:
        return X
    if rng is None:
        return X[:, CENTER_CROP]
    return X[np.arange(len(X)), rng.integers(0, X.shape[1], len(X))]


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    X = _crop(X, None)
    out = [torch.softmax(model(torch.from_numpy(X[i : i + batch_size])), 1).numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def fit(model: nn.Module, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
        n_classes: int, cfg: TrainConfig = TrainConfig(), run_name: str | None = None, verbose: bool = False) -> dict:
    """Train in place; restores the best-validation-macro-F1 weights. Returns history."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    alpha = class_weights(y_tr, n_classes)
    y_tr_t = torch.from_numpy(y_tr).long()

    best, best_state, bad = -1.0, None, 0
    history = {"loss": [], "val_macro_f1": []}
    start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        Xe = torch.from_numpy(augment_gain(_crop(X_tr, rng), rng))
        order = torch.randperm(len(Xe))
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            loss = focal_loss(model(Xe[idx]), y_tr_t[idx], alpha, cfg.focal_gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_f1 = f1_score(y_val, predict_proba(model, X_val).argmax(1), average="macro")
        history["loss"].append(total / len(order))
        history["val_macro_f1"].append(val_f1)
        if verbose:
            print(f"epoch {epoch:2d} loss {total / len(order):.4f} val macro-F1 {val_f1:.3f}")
        if val_f1 > best:
            best, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    history.update(best_val_macro_f1=best, epochs_run=epoch + 1, seconds=time.time() - start)
    _log_mlflow(run_name, cfg, history, model)
    return history


def _log_mlflow(run_name, cfg, history, model) -> None:
    """Experiment tracking is best-effort: training never fails because of MLflow."""
    if run_name is None:
        return
    try:
        import mlflow

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(asdict(cfg))
            mlflow.log_param("trainable_params", sum(p.numel() for p in model.parameters() if p.requires_grad))
            for step, (loss, f1) in enumerate(zip(history["loss"], history["val_macro_f1"])):
                mlflow.log_metrics({"train_loss": loss, "val_macro_f1": f1}, step=step)
            mlflow.log_metric("best_val_macro_f1", history["best_val_macro_f1"])
    except Exception as exc:  # pragma: no cover
        print(f"[mlflow] skipped: {exc}")
