#!/usr/bin/env python3
"""
Small-scale experiment on complexity dynamics of grokking.
Inspired by DeMoss et al., "The Complexity Dynamics of Grokking", arXiv:2412.09810
https://arxiv.org/abs/2412.09810

Tracks spectral entropy of weight matrices and a quantize+gzip compressed size proxy
while training a modular addition MLP with optional weight decay and spectral entropy regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gzip
import io

def spectral_entropy(weight: torch.Tensor) -> float:
    """H_svd = -sum bar_sigma log bar_sigma where bar_sigma are normalized singular values."""
    if weight.ndim != 2 or min(weight.shape) < 2:
        return 0.0
    try:
        s = torch.linalg.svdvals(weight)
        s = s[s > 1e-8]
        if len(s) == 0:
            return 0.0
        s_norm = s / s.sum()
        h = -torch.sum(s_norm * torch.log(s_norm + 1e-12))
        return h.item()
    except Exception:
        return 0.0

def model_spectral_entropy(model: nn.Module) -> float:
    total, count = 0.0, 0
    for param in model.parameters():
        if param.ndim == 2:
            total += spectral_entropy(param)
            count += 1
    return total / max(count, 1)

def compress_size(model: nn.Module, quant_bits: int = 6) -> int:
    """Proxy for apparent complexity: quantize all params then gzip."""
    params = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
    arr = np.concatenate(params)
    minv, maxv = arr.min(), arr.max()
    if maxv == minv:
        return 0
    levels = 2**quant_bits - 1
    q = np.round((arr - minv) / (maxv - minv) * levels).astype(np.uint8)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
        f.write(q.tobytes())
    return len(buf.getvalue())

class ModularAddMLP(nn.Module):
    def __init__(self, p: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(p, hidden)
        self.fc1 = nn.Linear(2 * hidden, hidden)
        self.fc2 = nn.Linear(hidden, p)
        self.act = nn.GELU()

    def forward(self, x):
        a, b = x[:, 0].long(), x[:, 1].long()
        ea, eb = self.embed(a), self.embed(b)
        h = self.act(self.fc1(torch.cat([ea, eb], dim=1)))
        return self.fc2(h)

def generate_data(p: int, frac: float, seed: int = 0):
    pairs = [(a, b) for a in range(p) for b in range(p)]
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_train = int(len(pairs) * frac)
    train = pairs[:n_train]
    test = pairs[n_train:]
    def to_tensors(data):
        x = torch.tensor([[a, b] for a, b in data], dtype=torch.float32)
        y = torch.tensor([(a + b) % p for a, b in data], dtype=torch.long)
        return x, y
    return to_tensors(train), to_tensors(test)

if __name__ == "__main__":
    print("Spectral entropy / compression proxy experiment for grokking")
    print("Paper: https://arxiv.org/abs/2412.09810")
    # Example short run configuration can be added here
    # Full training loops are in the daily notes
