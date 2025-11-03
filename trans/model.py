from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn

# ------------------------------
# Small from-scratch building blocks
# ------------------------------

def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    # Simple truncated normal init using rejection sampling (sufficient for small params)
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size).normal_(mean=mean, std=std)
        # clamp by resampling a few times (cheap & ok for init)
        for _ in range(4):
            mask = (tmp < a) | (tmp > b)
            if not mask.any():
                break
            tmp[mask] = torch.randn_like(tmp[mask]) * std + mean
        tensor.copy_(tmp)
        return tensor

def gelu(x: torch.Tensor) -> torch.Tensor:
    # Approximate GELU
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    y = torch.exp(x - m)
    z = y.sum(dim=dim, keepdim=True)
    return y / z

class Linear(nn.Module):
    """A no-bias Linear layer implemented with a single weight Parameter."""
    def __init__(self, in_features: int, out_features: int, *, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        trunc_normal_(self.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)  →  y: (..., out_features)
        return x.matmul(self.weight.t())

class Embedding(nn.Module):
    """Simple embedding: weight lookup by ids."""
    def __init__(self, num_embeddings: int, embedding_dim: int, *, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        trunc_normal_(self.weight, std=1.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class LayerNorm(nn.Module):
    """From-scratch LayerNorm with learnable gain/bias."""
    def __init__(self, d_model: int, eps: float = 1e-5, *, device=None, dtype=None):
        super().__init__()
        self.eps = float(eps)
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.gain + self.bias

class Dropout(nn.Module):
    """From-scratch inverted dropout (train only)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand_like(x) < keep).to(x.dtype)
        return x * mask / keep

# ------------------------------
# Config
# ------------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    ff_mult: int = 4
    seq_len: int = 256
    dropout: float = 0.1

# ------------------------------
# Attention
# ------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        self.drop = Dropout(dropout)
        # causal mask [1,1,L,L]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, seq_len, seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H, D = self.n_head, self.head_dim

        Q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)  # [B,H,L,D]
        K = self.k_proj(x).view(B, L, H, D).transpose(1, 2)  # [B,H,L,D]
        V = self.v_proj(x).view(B, L, H, D).transpose(1, 2)  # [B,H,L,D]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)     # [B,H,L,L]
        scores = scores.masked_fill(~self.mask[:, :, :L, :L], float("-inf"))
        P = softmax(scores, dim=-1)
        P = self.drop(P)
        Y = P @ V                                             # [B,H,L,D]
        Y = Y.transpose(1, 2).contiguous().view(B, L, C)      # merge heads
        return self.o_proj(Y)

# ------------------------------
# Transformer block
# ------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.seq_len, cfg.dropout)
        self.ln2 = LayerNorm(cfg.d_model)
        self.ff_in = Linear(cfg.d_model, cfg.ff_mult * cfg.d_model)
        self.ff_out = Linear(cfg.ff_mult * cfg.d_model, cfg.d_model)
        self.drop = Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        x = x + self.attn(self.ln1(x))
        h = gelu(self.ff_in(self.ln2(x)))
        h = self.drop(self.ff_out(h))
        return x + h

# ------------------------------
# GPT model
# ------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        trunc_normal_(self.pos_emb, std=0.02)
        self.drop = Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model)
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :L, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)
