"""Neural models: the CSI encoder (RuView-encoder stand-in), classifier and CNN-LSTM baseline.

The encoder uses explicit q/k/v/o and ff1/ff2 Linear layers so PEFT can inject LoRA
matrices by module name, and plain matmul attention so it exports cleanly to ONNX.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.heads, self.head_dim).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, 2 * dim)
        self.ff2 = nn.Linear(2 * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.ff2(F.gelu(self.ff1(self.ln2(x)))))


class CSIEncoder(nn.Module):
    """(B, T=100, 52) cleaned CSI amplitude -> (B, 128) embedding.

    Conv stem (2 x stride 2: 100 frames -> 25 tokens) -> learned positional embedding ->
    3 pre-norm transformer blocks (d=96) -> LayerNorm -> mean pooling -> Linear 96->128.
    Input normalisation is stored as buffers so the exported model takes cleaned
    amplitude directly. Sized to train on a laptop CPU; RuView's encoder replaces it.
    """

    def __init__(self, n_sub: int = 52, seq_len: int = 100, dim: int = 96, depth: int = 3,
                 heads: int = 4, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("norm_mean", torch.zeros(n_sub))
        self.register_buffer("norm_std", torch.ones(n_sub))
        self.stem = nn.Sequential(
            nn.Conv1d(n_sub, dim, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(dim, dim, 3, stride=2, padding=1), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.randn(1, seq_len // 4, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, embed_dim)
        self.dim = embed_dim

    def set_normalization(self, mean, std) -> None:
        self.norm_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.norm_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        x = self.stem(x.transpose(1, 2)).transpose(1, 2) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.proj(self.norm(x).mean(dim=1))


class CSIClassifier(nn.Module):
    def __init__(self, n_classes: int, **encoder_kwargs):
        super().__init__()
        self.encoder = CSIEncoder(**encoder_kwargs)
        self.head = nn.Linear(self.encoder.dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CNNLSTM(nn.Module):
    """M4: 1D-CNN + LSTM trained from scratch (tests the value of pre-training)."""

    def __init__(self, n_classes: int, n_sub: int = 52, hidden: int = 64):
        super().__init__()
        self.register_buffer("norm_mean", torch.zeros(n_sub))
        self.register_buffer("norm_std", torch.ones(n_sub))
        self.conv = nn.Sequential(
            nn.Conv1d(n_sub, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def set_normalization(self, mean, std) -> None:
        self.norm_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.norm_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
