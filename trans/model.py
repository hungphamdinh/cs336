from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

from trans.attention.flashAttention import can_use_flash_attention, flash_attention_forward
from trans.attention.localAttention import local_attention_forward

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
    """
    Approximate GELU using PyTorch's optimized implementation.
    This uses the tanh-based approximation, which is standard for Transformers
    and maps efficiently to Apple's AMX/bfloat16 on M1/M2.
    """
    return F.gelu(x, approximate="tanh")

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Wrapper around torch.softmax, which uses optimized kernels on each backend.
    """
    return torch.softmax(x, dim=dim)


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
    """
    LayerNorm wrapper that delegates to nn.LayerNorm.

    This leverages PyTorch's fused, backend-optimized implementation, which is
    better tuned for MPS/Apple Silicon than a from-scratch version using pow/sqrt.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, *, device=None, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)

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

        # Project to Q, K, V and shape as [B, H, L, D]
        Q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)  # [B,H,L,D]
        K = self.k_proj(x).view(B, L, H, D).transpose(1, 2)  # [B,H,L,D]
        V = self.v_proj(x).view(B, L, H, D).transpose(1, 2)  # [B,H,L,D]

        # First try FlashAttention-2 (on supported NVIDIA GPUs), otherwise use the local SDPA/manual path.
        Y = flash_attention_forward(
            Q,
            K,
            V,
            dropout_p=self.drop.p,
            training=self.training,
        )

        if Y is None:
            # Fallback for Mac M1/M2, older GPUs, or when flash-attn isn't installed.
            causal_mask = self.mask[:, :, :L, :L]
            Y = local_attention_forward(Q, K, V, causal_mask, self.drop)

        # Merge heads back to [B, L, C]
        Y = Y.transpose(1, 2).contiguous().view(B, L, C)
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
