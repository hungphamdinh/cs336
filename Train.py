#!/usr/bin/env python3
"""
Minimal, faithful Transformer (Vaswani et al., 2017).
- Encoder–Decoder, N layers, post-LayerNorm (Add → Norm)
- Multi-Head Attention + position-wise FFN (ReLU)
- Sinusoidal positional encodings
Includes mask builders and a tiny smoke test.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Positional Encoding (sinusoidal)
# -------------------------
class PositionalEncoding(nn.Module):
    """Fixed sinusoidal PE.

    Shapes: x ∈ [B,T,D] → [B,T,D]
    PE[t,2i] = sin(t·10000^(-2i/D)),  PE[t,2i+1] = cos(t·10000^(-2i/D))
    """
    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1,max_len,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# -------------------------
# Scaled Dot-Product Attention
# -------------------------
class ScaledDotProductAttention(nn.Module):
    """Per-head attention.

    Q:[B,H,Tq,d], K:[B,H,Tk,d], V:[B,H,Tk,dv]; mask→[B,H,Tq,Tk]
    Returns (out:[B,H,Tq,dv], attn:[B,H,Tq,Tk]).
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,Tq,Tk]
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out, attn


# -------------------------
# Multi-Head Attention
# -------------------------
class MultiHeadAttention(nn.Module):
    """H parallel attentions over subspaces, then concat + linear.

    Inputs q,k,v:[B,T,D]; mask→[B,H,Tq,Tk]. Output:[B,T,D].
    """
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, 'd_model must be divisible by number of heads'
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.h, self.d_k).permute(0, 2, 1, 3)  # [B,H,T,d_k]

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * d)  # [B,T,D]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split(self.W_q(q))
        K = self._split(self.W_k(k))
        V = self._split(self.W_v(v))
        out, _ = self.attn(Q, K, V, attn_mask)
        out = self._merge(out)
        out = self.W_o(out)
        return self.dropout(out)


# -------------------------
# Position-wise Feed-Forward (ReLU)
# -------------------------
class PositionwiseFFN(nn.Module):
    """2-layer MLP per position: D→d_ff→D with ReLU + Dropout."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.constant_(self.lin1.bias, 0.0)
        nn.init.constant_(self.lin2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(F.relu(self.lin1(x))))


# -------------------------
# Sublayer (Residual + Dropout + LayerNorm) — Post-LN
# -------------------------
class SublayerConnection(nn.Module):
    """Dropout → Residual Add → LayerNorm (post-LN)."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer_out))


# -------------------------
# Encoder & Decoder Layers (post-LN)
# -------------------------
class EncoderLayer(nn.Module):
    """Self-Attn → FFN, each with Add&Norm."""
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.sublayer1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.sublayer2(x, ffn_out)
        return x


class DecoderLayer(nn.Module):
    """Masked Self-Attn → Cross-Attn → FFN, each with Add&Norm."""
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self_out = self.self_attn(x, x, x, tgt_mask)
        x = self.sublayer1(x, self_out)
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.sublayer2(x, cross_out)
        ffn_out = self.ffn(x)
        x = self.sublayer3(x, ffn_out)
        return x


# -------------------------
# Encoder / Decoder stacks
# -------------------------
class Encoder(nn.Module):
    """Embeddings + PE, then N×(Self-Attn → FFN)."""
    def __init__(self, vocab: int, d_model: int, N: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """Embeddings + PE, then N×(Masked Self-Attn → Cross-Attn → FFN)."""
    def __init__(self, vocab: int, d_model: int, N: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x


# -------------------------
# Full Transformer
# -------------------------
class Transformer(nn.Module):
    """Encoder–Decoder + generator linear head."""
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, h, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, h, d_ff, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab)
        nn.init.xavier_uniform_(self.generator.weight)
        nn.init.constant_(self.generator.bias, 0.0)

    def forward(
        self,
        src: torch.Tensor,      # [B,S]
        tgt_in: torch.Tensor,   # [B,T]
        src_mask: Optional[torch.Tensor],  # [B,1,1,S]
        tgt_mask: Optional[torch.Tensor],  # [B,1,T,T]
    ) -> torch.Tensor:
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt_in, enc_out, tgt_mask, src_mask)
        return self.generator(dec_out)  # [B,T,V]


# -------------------------
# Masks (True = keep, False = mask)
# -------------------------
def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B,1,1,S] keep-mask for encoder & cross-attn (pads masked)."""
    return (src != pad_id).unsqueeze(1).unsqueeze(1)


def make_subsequent_mask(T: int, device=None) -> torch.Tensor:
    """[1,1,T,T] lower-triangular causal keep-mask for decoder self-attn."""
    m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B,1,T,T] = padding mask ∧ causal mask for decoder self-attn."""
    B, T = tgt.shape
    pad = (tgt != pad_id).unsqueeze(1).unsqueeze(1)  # [B,1,1,T]
    causal = make_subsequent_mask(T, device=tgt.device)  # [1,1,T,T]
    return pad & causal


# -------------------------
# Tiny smoke test
# -------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    model = Transformer(97, 101, d_model=128, N=2, h=4, d_ff=256, dropout=0.1)
    src = torch.randint(0, 97, (3, 11))
    tgt_in = torch.randint(0, 101, (3, 9))
    src_mask = make_src_mask(src, pad_id=0)
    tgt_mask = make_tgt_mask(tgt_in, pad_id=0)
    out = model(src, tgt_in, src_mask, tgt_mask)
    print('logits:', out.shape)  # [3,9,101]