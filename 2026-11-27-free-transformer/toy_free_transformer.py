"""
Toy experiment inspired by Free Transformer (Fleuret, arXiv:2510.17558).

Synthetic task: sequences of bits where a latent mode Z ~ Bernoulli(0.5)
determines that each token is Z flipped with small probability epsilon.
Pure autoregressive model must infer Z post-hoc from noisy previous tokens.
Model with explicit latent can "decide" Z early (via encoder during training,
or sample during generation).

We compare:
1. Standard small causal Transformer (AR baseline)
2. Free-style: injects a discrete latent (sampled or encoded) into the middle layer

Metrics: next-token accuracy / NLL on held-out sequences, and ability to
generate consistent sequences (low entropy of inferred mode from samples).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os

# Config
SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 15
HIDDEN = 64
HEADS = 4
LAYERS = 4  # total for baseline; free uses half + injection
EPSILON = 0.1  # flip probability
LATENT_SIZE = 2  # binary mode, one-hot
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

torch.manual_seed(42)
np.random.seed(42)

def generate_data(n_samples, seq_len, epsilon):
    Z = torch.randint(0, 2, (n_samples, 1))  # latent mode per sequence
    noise = torch.rand(n_samples, seq_len) < epsilon
    tokens = Z.expand(-1, seq_len) ^ noise.long()  # XOR with noise
    # shift for AR: input is tokens[:, :-1], target tokens[:, 1:]
    return tokens

# Generate datasets
train_tokens = generate_data(8000, SEQ_LEN, EPSILON)
val_tokens = generate_data(1000, SEQ_LEN, EPSILON)

train_ds = TensorDataset(train_tokens[:, :-1], train_tokens[:, 1:])
val_ds = TensorDataset(val_tokens[:, :-1], val_tokens[:, 1:])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        # Causal mask
        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

class StandardTransformer(nn.Module):
    def __init__(self, vocab=2, hidden=HIDDEN, heads=HEADS, layers=LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN-1, hidden) * 0.02)
        self.blocks = nn.ModuleList([CausalTransformerBlock(hidden, heads) for _ in range(layers)])
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.head(x)

class FreeStyleTransformer(nn.Module):
    """Minimal Free Transformer: inject discrete latent in middle, with simple encoder for training."""
    def __init__(self, vocab=2, hidden=HIDDEN, heads=HEADS, layers=LAYERS, latent_size=LATENT_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN-1, hidden) * 0.02)
        half = layers // 2
        self.blocks1 = nn.ModuleList([CausalTransformerBlock(hidden, heads) for _ in range(half)])
        self.blocks2 = nn.ModuleList([CausalTransformerBlock(hidden, heads) for _ in range(layers - half)])
        # Encoder: non-causal on first half output to predict latent
        self.enc_block = nn.TransformerEncoderLayer(hidden, heads, batch_first=True, activation="gelu")
        self.enc_head = nn.Linear(hidden, latent_size)
        # Injection: project one-hot Z to hidden and add to residual
        self.z_proj = nn.Linear(latent_size, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.latent_size = latent_size

    def encode_latent(self, x_emb):
        # Simple: mean pool after non-causal layer
        y = self.enc_block(x_emb)
        logits = self.enc_head(y.mean(dim=1))  # global latent
        return logits

    def forward(self, x, z=None, train_mode=True):
        x_emb = self.embed(x) + self.pos[:, :x.size(1)]
        for block in self.blocks1:
            x_emb = block(x_emb)
        if train_mode and z is None:
            # Use encoder to sample Z consistent with sequence
            z_logits = self.encode_latent(x_emb.detach())  # stop grad for simplicity in toy
            z_prob = F.softmax(z_logits, dim=-1)
            z_idx = torch.multinomial(z_prob, 1).squeeze(-1)
            z = F.one_hot(z_idx, self.latent_size).float()
            # KL term would be here; for toy we skip explicit KL and just use it
        elif z is None:
            # Inference: sample uniform
            z_idx = torch.randint(0, self.latent_size, (x.size(0),), device=x.device)
            z = F.one_hot(z_idx, self.latent_size).float()
        z_add = self.z_proj(z).unsqueeze(1)  # broadcast over sequence
        x_mid = x_emb + z_add
        for block in self.blocks2:
            x_mid = block(x_mid)
        return self.head(x_mid), z

def train_model(model, name, is_free=False):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = {"train_nll": [], "val_nll": [], "val_acc": []}
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            if is_free:
                logits, _ = model(xb, train_mode=True)
            else:
                logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, 2), yb.view(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        history["train_nll"].append(total_loss / len(train_loader))

        # Val
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if is_free:
                    logits, _ = model(xb, train_mode=False)  # sample Z for val? or use encoder
                    # better: use encoder for consistent eval
                    logits, _ = model(xb, train_mode=True)
                else:
                    logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, 2), yb.view(-1))
                val_loss += loss.item()
                preds = logits.argmax(-1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        history["val_nll"].append(val_loss / len(val_loader))
        history["val_acc"].append(correct / total)
        print(f"{name} Epoch {epoch+1}: train NLL {history['train_nll'][-1]:.4f}, val NLL {history['val_nll'][-1]:.4f}, val acc {history['val_acc'][-1]:.4f}")
    return history

print("Training Standard Transformer...")
std_model = StandardTransformer()
std_hist = train_model(std_model, "Standard")

print("\nTraining Free-style Transformer...")
free_model = FreeStyleTransformer()
free_hist = train_model(free_model, "Free-style", is_free=True)

# Plot results
os.makedirs("/workspace/artifacts", exist_ok=True)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(std_hist["val_nll"], label="Standard AR")
plt.plot(free_hist["val_nll"], label="Free-style")
plt.xlabel("Epoch")
plt.ylabel("Val NLL")
plt.legend()
plt.title("Validation NLL")
plt.subplot(1, 2, 2)
plt.plot(std_hist["val_acc"], label="Standard AR")
plt.plot(free_hist["val_acc"], label="Free-style")
plt.xlabel("Epoch")
plt.ylabel("Val Accuracy")
plt.legend()
plt.title("Validation Accuracy")
plt.tight_layout()
plot_path = "/workspace/artifacts/free_transformer_toy_results.png"
plt.savefig(plot_path, dpi=150)
print(f"Saved plot to {plot_path}")

# Final numbers
print("\n=== Final Results ===")
print(f"Standard final val NLL: {std_hist['val_nll'][-1]:.4f}, acc: {std_hist['val_acc'][-1]:.4f}")
print(f"Free-style final val NLL: {free_hist['val_nll'][-1]:.4f}, acc: {free_hist['val_acc'][-1]:.4f}")

# Generate some sequences to show consistency
def generate_sequences(model, n=8, is_free=False):
    model.eval()
    seqs = []
    with torch.no_grad():
        for _ in range(n):
            seq = torch.randint(0, 2, (1, 1), device=DEVICE)
            for t in range(SEQ_LEN - 2):
                if is_free:
                    logits, z = model(seq, train_mode=False)
                else:
                    logits = model(seq)
                next_tok = torch.multinomial(F.softmax(logits[:, -1], dim=-1), 1)
                seq = torch.cat([seq, next_tok], dim=1)
            seqs.append(seq.cpu().squeeze().tolist())
    return seqs

print("\nSample generations (Standard):")
print(generate_sequences(std_model, is_free=False))
print("\nSample generations (Free-style):")
print(generate_sequences(free_model, is_free=True))
