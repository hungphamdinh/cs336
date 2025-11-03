"""
CS336 Assignment 1 (basics): Reference scaffold from scratch
-------------------------------------------------------------------------------
This single file provides a from-scratch reference implementation for the
major components requested by the assignment handout:

  • BPE tokenizer training + encoding/decoding (byte-level)
  • Core NN layers: Linear (no bias), Embedding, RMSNorm
  • SwiGLU position-wise feed-forward
  • Rotary Positional Embedding (RoPE)
  • Softmax, scaled dot-product attention with optional mask
  • Causal multi-head self-attention (with RoPE)
  • Pre-norm TransformerBlock
  • TransformerLM (decoder-only)
  • Cross-entropy loss, AdamW optimizer (decoupled) and simple training loop
  • Minimal checkpointing helpers

Notes
-----
1) This file is provided for learning and experimentation. You should adapt,
   profile, and test each part yourself. It is intentionally written in a clear,
   explicit style and avoids torch.nn.functional and torch.optim conveniences,
   consistent with the assignment’s spirit.
2) The public tests for the official assignment rely on an `adapters.py` glue
   file to call into *your* implementations. At the bottom of this file you will
   find example adapter-style functions you can copy into your own `adapters.py`
   and point at the classes/functions below.
3) For speed, consider replacing some Python-side loops in BPE with more
   optimized structures. The current implementation aims for correctness and
   clarity first.
4) RoPE uses the common θ = 10_000.0 default unless you pass a different one.

Python >= 3.10, PyTorch >= 2.2 recommended.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import math
import json
import os
import io
import base64
import regex as re  # fast, supports the GPT-2 style pattern

import torch
from torch import nn
import argparse
from pathlib import Path

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -3.0, b: float = 3.0):
    """Truncated normal init to match assignment guidance.
    Wrapper around nn.init.trunc_normal_.
    """
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def round_up_multiple(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# --------------------------------------------------------------------------------------
# Byte‑level BPE training + Tokenizer
# --------------------------------------------------------------------------------------

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@dataclass
class BPESerialized:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]

    def save(self, vocab_path: str, merges_path: str):
        # Store vocab as JSON with base-256 via latin1
        with open(vocab_path, "w", encoding="utf-8") as f:
            # int->str: we encode bytes via latin1 to preserve 0..255 maps
            json.dump({int(k): v.decode('latin1') for k, v in self.vocab.items()}, f)
        # Store merges safely as JSON list of base64-encoded pairs to avoid delimiter issues
        merges_json = [
            [base64.b64encode(a).decode("ascii"), base64.b64encode(b).decode("ascii")]
            for (a, b) in self.merges
        ]
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(merges_json, f)

    @staticmethod
    def load(vocab_path: str, merges_path: str) -> "BPESerialized":
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {int(k): v.encode('latin1') for k, v in vocab_json.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        merges: List[Tuple[bytes, bytes]] = []
        # Try new JSON format first
        try:
            data = json.loads(raw)
            for a_b64, b_b64 in data:
                merges.append((base64.b64decode(a_b64), base64.b64decode(b_b64)))
        except json.JSONDecodeError:
            # Legacy fallback: lines with two latin1 tokens separated by a single space.
            # WARNING: Unsafe if tokens themselves contain space; prefer retraining to regenerate JSON merges.
            for line in raw.splitlines():
                if not line:
                    continue
                parts = line.rstrip("\n").split(" ", 1)
                if len(parts) != 2:
                    raise ValueError(
                        "Failed to parse legacy merges format. "
                        "Please delete the merges file and retrain BPE to write JSON-format merges."
                    )
                a, b = parts
                merges.append((a.encode('latin1'), b.encode('latin1')))
        return BPESerialized(vocab=vocab, merges=merges)


def _split_on_special(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]
    # Build a regex that splits and keeps delimiters
    parts: List[str] = []
    # Escape specials and join with alternation
    specials_re = "|".join(re.escape(tok) for tok in special_tokens)
    pattern = re.compile(f"({specials_re})")
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append(text[last:m.start()])
        parts.append(m.group(0))
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return parts


def _pretokenize_iter(chunks: Iterable[str]) -> Dict[Tuple[bytes, ...], int]:
    """Count pre-token frequencies using GPT-2 regex; represent each as tuple of bytes.
    Returns dict mapping tuple(bytes) -> frequency.
    """
    counts: Dict[Tuple[bytes, ...], int] = {}
    pat = re.compile(GPT2_PAT)
    for chunk in chunks:
        for m in pat.finditer(chunk):
            token = m.group(0)
            b = token.encode("utf-8")
            key = tuple(bytes([x]) for x in b)  # each entry is a bytes of len=1
            counts[key] = counts.get(key, 0) + 1
    return counts


def _count_pairs(counts: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
    pair_counts: Dict[Tuple[bytes, bytes], int] = {}
    for word, c in counts.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + c
    return pair_counts


def _merge_once(
    counts: Dict[Tuple[bytes, ...], int],
    pair: Tuple[bytes, bytes],
) -> Dict[Tuple[bytes, ...], int]:
    """Return a new counts dict where every occurrence of `pair` is merged into `pair[0]+pair[1]`."""
    a, b = pair
    ab = a + b
    new_counts: Dict[Tuple[bytes, ...], int] = {}
    for word, c in counts.items():
        if len(word) < 2:
            new_counts[word] = new_counts.get(word, 0) + c
            continue
        merged: List[bytes] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                merged.append(ab)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        new_counts[tuple(merged)] = new_counts.get(tuple(merged), 0) + c
    return new_counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: Optional[List[str]] = None,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer (naïve but clear CPU implementation).

    Args
    ----
    input_path: path to text file
    vocab_size: total vocab size including initial 256 bytes and special tokens
    special_tokens: list of strings preserved as single tokens
    """
    special_tokens = special_tokens or []

    # 1) Initialize vocab as all bytes + special tokens at the end of id space
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # 2) Load corpus; split so merges never cross specials, then pretokenize
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    parts = _split_on_special(text, special_tokens)

    # Only feed the non-special parts to the regex pretokenizer
    non_special_chunks = (p for p in parts if p not in special_tokens)
    counts = _pretokenize_iter(non_special_chunks)

    merges: List[Tuple[bytes, bytes]] = []
    # 3) Iteratively merge the most frequent pair until we hit vocab_size
    while len(vocab) < vocab_size:
        pair_counts = _count_pairs(counts)
        if not pair_counts:
            break
        # pick by max frequency; break ties by lexicographically greater pair (as bytes)
        max_freq = max(pair_counts.values())
        candidates = [pair for pair, c in pair_counts.items() if c == max_freq]
        best = max(candidates)  # lexicographically greatest bytes pair

        counts = _merge_once(counts, best)
        merges.append(best)
        # add new token
        a, b = best
        vocab[len(vocab)] = a + b

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer with special-token support.

    Encoding uses the standard greedy BPE merges by pair-rank.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        self.id_to_token: Dict[int, bytes] = dict(vocab)
        self.token_to_id: Dict[bytes, int] = {v: k for k, v in self.id_to_token.items()}

        # Extend with special tokens if not present
        if special_tokens:
            for tok in special_tokens:
                b = tok.encode("utf-8")
                if b not in self.token_to_id:
                    new_id = len(self.id_to_token)
                    self.id_to_token[new_id] = b
                    self.token_to_id[b] = new_id

        # Merge ranks for greedy application
        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }
        self.special_tokens = set((special_tokens or []))
        self._pat = re.compile(GPT2_PAT)

    # ---- Construction helpers ----
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        bpe = BPESerialized.load(vocab_filepath, merges_filepath)
        return cls(bpe.vocab, bpe.merges, special_tokens=special_tokens)

    # ---- Core BPE encode/decode ----
    def _bpe_encode_bytes(self, b: bytes) -> List[int]:
        # Represent as list of byte-chunks initially (each of length 1)
        word: List[bytes] = [bytes([x]) for x in b]
        if not word:
            return []

        # Map pair -> rank for quick lookup
        ranks = self.merge_ranks

        # Precompute a function to find best merge index
        def get_pair_rank(i: int) -> int:
            if i < 0 or i >= len(word) - 1:
                return 10**12  # effectively inf rank
            return ranks.get((word[i], word[i + 1]), 10**12)

        # Build a linked-list-like structure of next/prev indices to update locally
        import heapq

        # heap of (rank, i). We'll lazily discard stale entries.
        heap: List[Tuple[int, int]] = []
        for i in range(len(word) - 1):
            heapq.heappush(heap, (get_pair_rank(i), i))

        while heap:
            rank, i = heapq.heappop(heap)
            if i >= len(word) - 1:
                continue
            if ranks.get((word[i], word[i + 1]), 10**12) != rank:
                # stale
                continue
            if rank >= 10**12:
                break
            # Merge at i
            merged = word[i] + word[i + 1]
            word[i : i + 2] = [merged]
            # Update neighbors around i
            if i - 1 >= 0:
                heapq.heappush(heap, (get_pair_rank(i - 1), i - 1))
            if i < len(word) - 1:
                heapq.heappush(heap, (get_pair_rank(i), i))

        # Map final byte-chunks to ids (must exist in vocab)
        ids: List[int] = []
        for chunk in word:
            tid = self.token_to_id.get(chunk)
            if tid is None:
                # Fallback: ensure every byte exists; split into raw bytes ids
                ids.extend(self.token_to_id[bytes([x])] for x in chunk)
            else:
                ids.append(tid)
        return ids

    def encode(self, text: str) -> List[int]:
        # Respect special tokens: split and keep them intact
        ids: List[int] = []
        parts = _split_on_special(text, list(self.special_tokens))
        for part in parts:
            if part in self.special_tokens:
                tid = self.token_to_id.get(part.encode("utf-8"))
                if tid is None:
                    # Should not happen; safety
                    tid = self.token_to_id[part.encode("utf-8")]
                ids.append(tid)
            else:
                for m in self._pat.finditer(part):
                    token = m.group(0)
                    ids.extend(self._bpe_encode_bytes(token.encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for tid in self.encode(s):
                yield tid

    def decode(self, ids: Sequence[int]) -> str:
        blob = b"".join(self.id_to_token.get(int(i), b"") for i in ids)
        # Replace malformed sequences by U+FFFD
        return blob.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------------------
# Core Layers: Linear (no bias), Embedding, RMSNorm
# --------------------------------------------------------------------------------------

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # He-like (fan-in+fan-out) trunc normal suggested by handout
        std = math.sqrt(2.0 / (in_features + out_features))
        trunc_normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # y = x @ W^T
        return x.matmul(self.weight.t())


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        trunc_normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = float(eps)
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f = x.to(torch.float32)
        rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_f / rms * self.gain
        return y.to(in_dtype)


# --------------------------------------------------------------------------------------
# Feed-Forward: SwiGLU
# --------------------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(round(8.0 * d_model / 3.0))
        d_ff = round_up_multiple(d_ff, 64)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.silu(self.w1(x))
        b = self.w3(x)
        return self.w2(a * b)


# --------------------------------------------------------------------------------------
# RoPE: Rotary Positional Embedding
# --------------------------------------------------------------------------------------

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE pairs"
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        # Precompute inv_freq per pair (size d_k/2)
        half = d_k // 2
        # inv_freq[k] = theta^{-2k/d_k}
        idx = torch.arange(0, half, dtype=torch.float32, device=device)
        inv_freq = self.theta ** (-2.0 * idx / self.d_k)
        # Precompute for positions 0..max_len-1: angles = pos[:,None] * inv_freq[None,:]
        pos = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        angles = pos[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq, d_k)
        *prefix, seq, d = x.shape
        assert d == self.d_k
        # gather cos/sin for positions
        cos = self.cos_cached[token_positions]  # (..., seq, half)
        sin = self.sin_cached[token_positions]
        # reshape x into pairs (..., seq, half, 2)
        x_ = x.view(*prefix, seq, d // 2, 2)
        x_even = x_[..., 0]
        x_odd = x_[..., 1]
        # rotate: [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]
        xo = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return xo.view(*prefix, seq, d)


# --------------------------------------------------------------------------------------
# Softmax + Scaled Dot-Product Attention
# --------------------------------------------------------------------------------------

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    y = torch.exp(x - m)
    z = y.sum(dim=dim, keepdim=True)
    return y / z


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute attention(Q,K,V) with optional boolean mask.

    Shapes (batch-like dims allowed):
        Q: (..., q_len, d_k)
        K: (..., k_len, d_k)
        V: (..., k_len, d_v)
        mask (optional): (q_len, k_len) with True = keep, False = block
    Returns:
        (..., q_len, d_v)
    """
    d_k = Q.shape[-1]
    # (..., q, k)
    scores = Q.matmul(K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Convert boolean (True=keep, False=block) to additive mask
        # Fill blocked entries with -inf to zero them after softmax
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.to(dtype=torch.bool), neg_inf)
    P = softmax(scores, dim=-1)
    return P.matmul(V)


# --------------------------------------------------------------------------------------
# Causal Multi-Head Self-Attention (with RoPE)
# --------------------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, rope_theta: float = 10_000.0, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.Wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        H = self.num_heads
        d = self.d_head

        def proj_and_reshape(W: Linear) -> torch.Tensor:
            y = W(x)  # (B, L, D)
            y = y.view(B, L, H, d).permute(0, 2, 1, 3)  # (B, H, L, d)
            return y

        Q = proj_and_reshape(self.Wq)
        K = proj_and_reshape(self.Wk)
        V = proj_and_reshape(self.Wv)

        # Apply RoPE to Q,K per head (treat head as batch-like)
        pos = torch.arange(L, device=x.device).view(1, 1, L).expand(B, H, L)
        Q = self.rope(Q, pos)
        K = self.rope(K, pos)

        # Causal mask: (L, L) where i attends to j <= i
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))

        # Compute attention per (B,H) batch
        attn_out = scaled_dot_product_attention(Q, K, V, mask=causal)  # (B, H, L, d)
        # Merge heads
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.Wo(attn_out)


# --------------------------------------------------------------------------------------
# Transformer Block and Transformer LM
# --------------------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, d_ff: Optional[int] = None, device=None, dtype=None):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        context_length: int,
        d_ff: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.tok = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, max_seq_len=context_length, d_ff=d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L)
        x = self.tok(token_ids)  # (B, L, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, L, V)
        return logits


# --------------------------------------------------------------------------------------
# Loss, Optimizer (AdamW), and Training Loop
# --------------------------------------------------------------------------------------

def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    y = (x - m).exp().sum(dim=dim, keepdim=True).log() + m
    return y


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean token-level NLL.
    logits: (B, L, V); targets: (B, L) long
    """
    logZ = logsumexp(logits, dim=-1).squeeze(-1)
    # gather logit of target index
    tgt_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    nll = (logZ - tgt_logits).mean()
    return nll


class AdamW:
    """Minimal decoupled AdamW (no torch.optim usage).

    You can wrap this with an interface similar to torch.optim:
        opt = AdamW(model.parameters(), lr=3e-4)
        opt.step(); opt.zero_grad()
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.state: Dict[int, Dict[str, torch.Tensor | int]] = {}

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            sid = id(p)
            st = self.state.get(sid)
            if st is None:
                st = {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}
                self.state[sid] = st
            st["t"] += 1
            m = st["m"]
            v = st["v"]
            t = st["t"]

            # Adam moments
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Decoupled weight decay
            if self.weight_decay != 0.0:
                p.data.add_(p.data, alpha=-self.weight_decay * self.lr)

            # Parameter update
            p.data.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)


# Simple toy dataloader for next-token LM from a flat token tensor

class LMSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids: torch.Tensor, context_length: int):
        self.tokens = token_ids.to(torch.long)
        self.L = context_length
        # We form sequences of length L where target is next token; we discard last tail
        self.n = (len(self.tokens) - 1) // self.L

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.L
        x = self.tokens[s : s + self.L]
        y = self.tokens[s + 1 : s + 1 + self.L]
        return x, y


# --------------------------------------------------------------------------------------
# TinyStories tokenizer/encoding helpers
# --------------------------------------------------------------------------------------
def encode_text_file_to_tensor(txt_path: str, tokenizer: Tokenizer) -> torch.Tensor:
    """
    Stream-encode a UTF-8 text file to a flat 1-D LongTensor of token ids.
    This will read line-by-line to keep memory bounded during tokenization.
    """
    ids: List[int] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            ids.extend(tokenizer.encode(line))
    if not ids:
        # Ensure at least one token to avoid edge-cases in dataset math
        ids = [0]
    return torch.tensor(ids, dtype=torch.long)

def build_loaders(train_ids: torch.Tensor, valid_ids: torch.Tensor, context_length: int, batch_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_ds = LMSequenceDataset(train_ids, context_length)
    valid_ds = LMSequenceDataset(valid_ids, context_length)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, valid_loader


# --------------------------------------------------------------------------------------
# Train on TinyStories-style text files
# --------------------------------------------------------------------------------------
def train_tinystories(
    train_txt: str,
    valid_txt: str,
    out_dir: str,
    *,
    vocab_size: int = 4096,
    context_length: int = 256,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 8,
    d_ff: int | None = None,
    batch_size: int = 64,
    epochs: int = 1,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    eval_every: int = 500,
    save_every: int = 2000,
    seed: int = 1337,
):
    """
    End-to-end training on two plain-text corpora (train/valid) using:
      - byte-level BPE (trained on train_txt)
      - TransformerLM (decoder-only) with SwiGLU/RMSNorm/RoPE
    Artifacts saved to out_dir: vocab.json, merges.txt, *.pt checkpoints.
    """
    torch.manual_seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab_path = str(out / "bpe_vocab.json")
    merges_path = str(out / "bpe_merges.txt")
    ckpt_path = lambda step: str(out / f"ckpt_step{step}.pt")

    special_tokens = ["<|story_start|>", "<|story_end|>", "<|endoftext|>"]

    # --- Train BPE on training text (byte-level GPT-2 pattern) ---
    print(f"[BPE] training on: {train_txt}  → vocab_size={vocab_size}")
    vocab, merges = train_bpe(train_txt, vocab_size=vocab_size, special_tokens=special_tokens)
    # Save vocab/merges for reuse
    BPESerialized(vocab=vocab, merges=merges).save(vocab_path, merges_path)
    print(f"[BPE] saved to: {vocab_path} / {merges_path}")

    # --- Build tokenizer from saved files (mirrors real workflow) ---
    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)

    # --- Encode train/valid to flat token tensors ---
    print("[ENC] encoding train…")
    train_ids = encode_text_file_to_tensor(train_txt, tok)
    print(f"[ENC] train tokens: {len(train_ids)}")
    print("[ENC] encoding valid…")
    valid_ids = encode_text_file_to_tensor(valid_txt, tok)
    print(f"[ENC] valid tokens: {len(valid_ids)}")

    # --- DataLoaders ---
    train_loader, valid_loader = build_loaders(train_ids, valid_ids, context_length, batch_size)

    # --- Model/opt ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        context_length=context_length,
        d_ff=d_ff,
        device=device,
        dtype=torch.float32,
    ).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Train ---
    global_step = 0
    best_val_ppl = float("inf")
    print(f"[RUN] device={device}, steps/epoch≈{len(train_loader)}, eval_every={eval_every}, save_every={save_every}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for it, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            total_loss += float(loss.item())
            global_step += 1

            if global_step % 100 == 0:
                avg = total_loss / 100
                print(f"[epoch {epoch}] step {global_step:>6}  train_loss={avg:.4f}")
                total_loss = 0.0

            if global_step % eval_every == 0:
                ppl = estimate_perplexity(model, valid_loader, device)
                print(f"[eval] step {global_step}  valid_ppl={ppl:.2f}")
                if ppl < best_val_ppl:
                    best_val_ppl = ppl
                    path = ckpt_path(global_step)
                    save_checkpoint(path, model, opt, global_step)
                    print(f"[ckpt] saved best @ step {global_step} → {path}")

            if global_step % save_every == 0:
                path = ckpt_path(global_step)
                save_checkpoint(path, model, opt, global_step)
                print(f"[ckpt] periodic save @ step {global_step} → {path}")

        # end epoch
        ppl = estimate_perplexity(model, valid_loader, device)
        print(f"[epoch {epoch} end] valid_ppl={ppl:.2f}")
        if ppl < best_val_ppl:
            best_val_ppl = ppl
            path = ckpt_path(global_step)
            save_checkpoint(path, model, opt, global_step)
            print(f"[ckpt] saved best @ step {global_step} → {path}")

    print(f"[DONE] best_valid_ppl={best_val_ppl:.2f}  artifacts in: {out_dir}")


@torch.no_grad()
def estimate_perplexity(model: TransformerLM, data: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in data:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        total_loss += float(loss.item()) * x.shape[0]
        count += x.shape[0]
    model.train()
    mean_loss = total_loss / max(count, 1)
    return float(math.exp(mean_loss))


def save_checkpoint(path: str, model: nn.Module, opt: AdamW, step: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": {
            "state": {k: {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()} for k, v in opt.state.items()},
            "lr": opt.lr,
            "beta1": opt.beta1,
            "beta2": opt.beta2,
            "eps": opt.eps,
            "weight_decay": opt.weight_decay,
        },
        "step": step,
    }, path)


def load_checkpoint(path: str, model: nn.Module, opt: Optional[AdamW] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"]) 
    if opt is not None:
        st = ckpt["opt"]
        opt.lr = st["lr"]
        opt.beta1 = st["beta1"]
        opt.beta2 = st["beta2"]
        opt.eps = st["eps"]
        opt.weight_decay = st["weight_decay"]
        # tensor states moved to correct device on first step
        opt.state = {int(k): {kk: vv for kk, vv in v.items()} for k, v in st["state"].items()}
    return ckpt.get("step", 0)


# --------------------------------------------------------------------------------------
# Example adapters glue (copy to adapters.py in your project and point to your code)
# --------------------------------------------------------------------------------------

# The real assignment expects adapters.py to *call into* your implementation.
# Here are minimal functions that match the names used by the tests in the handout.

# BPE training

def run_train_bpe(input_path: str, vocab_size: int, special_tokens: Optional[List[str]] = None):
    return train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)

# Tokenizer access for tests

def get_tokenizer(vocab_path: str, merges_path: str, special_tokens: Optional[List[str]] = None) -> Tokenizer:
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)

# Linear

def run_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    lin = Linear(weight.shape[1], weight.shape[0], device=x.device, dtype=x.dtype)
    lin.load_state_dict({"weight": weight})
    return lin(x)

# Embedding

def run_embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    emb = Embedding(weight.shape[0], weight.shape[1], device=ids.device, dtype=weight.dtype)
    emb.load_state_dict({"weight": weight})
    return emb(ids)

# RMSNorm

def run_rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm = RMSNorm(x.shape[-1], eps=eps, device=x.device, dtype=x.dtype)
    # gain initialized to 1.0 by default
    return norm(x)

# Softmax

def run_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return softmax(x, dim)

# Scaled dot-product attention

def run_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return scaled_dot_product_attention(Q, K, V, mask=mask)

# RoPE

def run_rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    rope = RotaryPositionalEmbedding(theta=theta, d_k=x.shape[-1], max_seq_len=int(positions.max().item()) + 1, device=x.device)
    return rope(x, positions)

# Multi-head self-attention

def run_multihead_self_attention(x: torch.Tensor, num_heads: int, max_seq_len: int) -> torch.Tensor:
    mha = MultiHeadSelfAttention(d_model=x.shape[-1], num_heads=num_heads, max_seq_len=max_seq_len, device=x.device, dtype=x.dtype)
    return mha(x)

# Transformer block

def run_transformer_block(x: torch.Tensor, num_heads: int, max_seq_len: int, d_ff: Optional[int] = None) -> torch.Tensor:
    blk = TransformerBlock(d_model=x.shape[-1], num_heads=num_heads, max_seq_len=max_seq_len, d_ff=d_ff, device=x.device, dtype=x.dtype)
    return blk(x)

# Transformer LM

def run_transformer_lm(token_ids: torch.Tensor, vocab_size: int, d_model: int, num_heads: int, num_layers: int, context_length: int, d_ff: Optional[int] = None) -> torch.Tensor:
    model = TransformerLM(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads, num_layers=num_layers, context_length=context_length, d_ff=d_ff, device=token_ids.device, dtype=torch.float32)
    return model(token_ids)



# --------------------------------------------------------------------------------------
# CLI for smoke test or TinyStories training
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 demo or TinyStories training")
    sub = parser.add_subparsers(dest="cmd", required=False)

    demo_p = sub.add_parser("demo", help="run minimal tokenizer+LM smoke test")
    tinyp = sub.add_parser("train", help="train on TinyStories-style text files")

    # TinyStories training args
    tinyp.add_argument("--train_txt", type=str, required=True, help="path to tinystories_train.txt")
    tinyp.add_argument("--valid_txt", type=str, required=True, help="path to tinystories_valid.txt")
    tinyp.add_argument("--out_dir", type=str, required=True, help="where to save BPE + checkpoints")
    tinyp.add_argument("--vocab_size", type=int, default=4096)
    tinyp.add_argument("--context_length", type=int, default=256)
    tinyp.add_argument("--d_model", type=int, default=512)
    tinyp.add_argument("--heads", type=int, default=8)
    tinyp.add_argument("--layers", type=int, default=8)
    tinyp.add_argument("--d_ff", type=int, default=None)
    tinyp.add_argument("--batch_size", type=int, default=64)
    tinyp.add_argument("--epochs", type=int, default=1)
    tinyp.add_argument("--lr", type=float, default=3e-4)
    tinyp.add_argument("--weight_decay", type=float, default=0.0)
    tinyp.add_argument("--eval_every", type=int, default=500)
    tinyp.add_argument("--save_every", type=int, default=2000)
    tinyp.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    if args.cmd == "train":
        train_tinystories(
            train_txt=args.train_txt,
            valid_txt=args.valid_txt,
            out_dir=args.out_dir,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_heads=args.heads,
            num_layers=args.layers,
            d_ff=args.d_ff,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            save_every=args.save_every,
            seed=args.seed,
        )
    else:
        # Default: run the original minimal smoke test
        torch.manual_seed(0)
        vocab = {i: bytes([i]) for i in range(256)}
        merges: List[Tuple[bytes, bytes]] = []
        tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
        s = "hello <|endoftext|> world"
        ids = tok.encode(s)
        print("Encoded:", ids)
        print("Decoded:", tok.decode(ids))

        B, L, V = 2, 8, 256
        model = TransformerLM(vocab_size=V, d_model=64, num_heads=4, num_layers=2, context_length=L)
        x = torch.randint(0, V, (B, L))
        logits = model(x)
        y = x.roll(-1, dims=1)
        loss = cross_entropy_loss(logits, y)
        print("CE loss:", float(loss))

        opt = AdamW(model.parameters(), lr=1e-3)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print("Step ok.")
