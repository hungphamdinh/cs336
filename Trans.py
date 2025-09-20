#!/usr/bin/env python3
"""
A minimal, faithful implementation of the *original* Transformer (Vaswani et al., 2017)
— focusing on the core model only (no datasets/training boilerplate).

Design choices match the paper's default setup:
- Encoder–Decoder with N identical layers
- Post-LayerNorm (Add & Norm after residual addition)
- Multi-Head Attention with scaled dot-product
- Position-wise FFN with ReLU
- Sinusoidal positional encodings
- Dropout everywhere the paper places it

This file exposes:
- class Transformer
- helper mask builders: make_src_mask, make_tgt_mask

Example
-------
>>> model = Transformer(src_vocab=32000, tgt_vocab=32000, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1)
>>> src = torch.randint(0, 32000, (2, 20))  # [B,T]
>>> tgt_in = torch.randint(0, 32000, (2, 18))  # [B,T]
>>> src_mask = make_src_mask(src, pad_id=0)   # [B,1,1,S]
>>> tgt_mask = make_tgt_mask(tgt_in, pad_id=0) # [B,1,T,T]
>>> logits = model(src, tgt_in, src_mask, tgt_mask)  # [B,T,C]
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Quick data-flow cheat sheet (Encoder–Decoder)
# -----------------------------------------------------------------------------
# src ids [B,S] --(Embedding+PE)--> [B,S,D] --(N× EncoderLayer)--> enc_out [B,S,D]
# tgt ids [B,T] --(Embedding+PE)--> [B,T,D] --(N× DecoderLayer with masks & cross-attn)--> dec_out [B,T,D]
# dec_out --(Linear generator)--> logits over tgt vocab [B,T,V]
#
# Masks
#  - src_mask: [B,1,1,S]  (True=keep, False=mask-pad). Broadcasts to [B,H,T_q,S] in cross-attn.
#  - tgt_mask: [B,1,T,T]  (padding AND causal). Broadcasts to [B,H,T,T] in decoder self-attn.
# =============================================================================

"""
# =============================================================================
# END-TO-END WALKTHROUGH — "Alice eats two apples"
# -----------------------------------------------------------------------------
# Goal: Show the *entire* forward pass on a tiny toy example so shapes & roles
# are crystal clear. Numbers below are illustrative; shapes are exact.
#
# Setup (toy):
#   B=1, D=4, S=6, T=6
#   Vocab slice: 0:<pad> 2:<bos> 3:<eos> 4:Alice 5:eats 6:two 7:apples
#   src = "<bos> Alice eats two apples <eos>"
#   tgt = same sentence (copy task), but the decoder is fed a *shifted* prefix
#        during training (teacher forcing).
#
# ────────────────────────────────────────────────────────────────────────────
# 1) Encoder
# ────────────────────────────────────────────────────────────────────────────
# 1.1 IDs → Embeddings (content)
#   src_ids = [2, 4, 5, 6, 7, 3] ∈ ℤ^{[1,6]}
#   x = Embedding(src_ids) ∈ ℝ^{[1,6,D]}  (gather rows from the embedding table)
#   Example x[0]:
#     [ 0.50, -0.20,  0.10,  0.30]   # <bos>
#     [ 0.20, -0.10,  0.30,  0.50]   # Alice
#     [ 0.00,  0.20, -0.10,  0.10]   # eats
#     [-0.20,  0.10,  0.10, -0.10]   # two
#     [ 0.10,  0.10,  0.20, -0.20]   # apples
#     [ 0.40,  0.05, -0.15,  0.00]   # <eos>
#   (Your code then scales by √D and adds sinusoidal PE; shape stays [1,6,D].)
#
# 1.2 Add Positional Encoding (order)
#   x ← x + PE[:,:S]  (then dropout). Same shape [1,6,D]. Now each token carries
#   *what* (embedding) + *where* (position) so attention can reason about both.
#
# 1.3 Encoder layers: Self-Attn → Add&Norm → FFN → Add&Norm (×N)
#   • Self-Attn: q=k=v=x → per-head softmax over the 6 tokens.
#     Example head behaviors (intuition):
#       – Head A (roles):   query="eats" attends to subject "Alice" and object "apples".
#       – Head B (local):   query="apples" attends to nearby quantifier "two".
#     Heads produce weighted sums; concat → W_o → dropout → Add&Norm.
#   • FFN (per position): D→d_ff→D with ReLU + dropout; adds nonlinearity to recompose
#     features *within* each token; Add&Norm again.
#   Result: enc_out ∈ ℝ^{[1,6,D]} where each source position is context-enriched.
#   Example: enc_out["apples"] carries identity (noun), plurality (from "two"), role (object of "eats").
#
# ────────────────────────────────────────────────────────────────────────────
# 2) Decoder (teacher forcing during training)
# ────────────────────────────────────────────────────────────────────────────
# 2.1 Build masks
#   src_mask: [1,1,1,S] (True on non-pad). Used by cross-attn to ignore encoder pads.
#   tgt_mask: [1,1,T-1,T-1] (padding AND *causal*). Causal mask is lower-triangular so
#             step t can attend only to ≤ t.
#
# 2.2 Decoder input embeddings + PE
#   We feed a *shifted* prefix (drop the final token):
#     tgt_in = ["<bos>", "Alice", "eats", "two", "apples"]  → [1,5]
#   x_dec = Embedding(tgt_in) * √D; add PE → [1,5,D].
#
# 2.3 Decoder layers: Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm (×N)
#   a) Masked Self-Attn (q=k=v=x_dec; masked by tgt_mask)
#      Each position can only look leftward (no peeking). For the "apples" position,
#      it can attend to "two" in the prefix and carry plurality within the decoder stream.
#   b) Cross-Attn (q=x_dec, k=v=enc_out)
#      Alignment/retrieval from the source. Example for predicting "apples":
#        – Head A (quantity ↔ noun proximity): weights over src
#          [bos, Alice, eats, two, apples, eos] → [0.05, 0.05, 0.10, 0.60, 0.15, 0.05]
#        – Head B (lexical/object identity):    → [0.02, 0.15, 0.10, 0.08, 0.60, 0.05]
#        z_A,z_B are concatenated and mixed by W_o. Separate channels keep count and object
#        evidence disentangled *before* fusion.
#   c) FFN per position to nonlinearly recompose features for prediction.
#   Result: dec_out ∈ ℝ^{[1,5,D]}.
#
# ────────────────────────────────────────────────────────────────────────────
# 3) Generator → logits (and probabilities)
# ────────────────────────────────────────────────────────────────────────────
#   logits = dec_out @ W_gen + b ∈ ℝ^{[1,5,V]}
#   Training: CrossEntropyLoss(logits[:,t,:], gold_next_id[t]) for t=0..4
#   Inference: softmax(logits[:,-1,:]) → pick next token (greedy/beam/sampling);
#              append to prefix; repeat until <eos>.
#
# ────────────────────────────────────────────────────────────────────────────
# 4) Quick shape recap (one pass)
# ────────────────────────────────────────────────────────────────────────────
#   Encoder
#     src_ids        : [1, 6]
#     embed+PE       : [1, 6, D]
#     N×EncoderLayer : [1, 6, D]  → enc_out
#   Decoder (teacher forcing)
#     tgt_in_ids     : [1, 5]
#     embed+PE       : [1, 5, D]
#     masked SA      : [1, 5, D]
#     cross-attn     : [1, 5, D]
#     FFN            : [1, 5, D]  → dec_out
#   Generator
#     logits         : [1, 5, V]
# =============================================================================
"""
# -------------------------
# Positional Encoding (sinusoidal)
# -------------------------
class PositionalEncoding(nn.Module):
    """
    Fixed (non-learned) sinusoidal positional encoding from Vaswani et al. (2017).

    Shapes
    -------
    Input  x: [B, T, D]  (batch, time/sequence length, model dim)
    Buffer pe: [1, max_len, D]
    Output   : [B, T, D]

    Definition
    ----------
    For position t and channel pair i (0-based):
        PE[t, 2i]   = sin( t * 10000^(-2i/D) )
        PE[t, 2i+1] = cos( t * 10000^(-2i/D) )
    Frequencies are geometrically spaced so some channels vary quickly (local),
    others slowly (global). Sin/cos pairs make relative shifts a linear transform.

    Notes
    -----
    * This exact implementation assumes D is even (pairs of sin/cos). If D is odd,
      see the comment near the cosine assignment for a compatible tweak.
    * `pe` is registered as a *buffer* (not a Parameter): it moves with the model
      and is checkpointed, but it is not trainable.
    """
    def __init__(self, d_model: int, max_len: int = 10_000, dropout: float = 0.1):
        super().__init__()  # Initialize nn.Module internals (state dict, device moves, etc.)

        # Dropout applied *after* adding positions to token embeddings
        self.dropout = nn.Dropout(dropout)

        # Allocate the positional table (to be filled with sin/cos values)
        pe = torch.zeros(max_len, d_model)  # [max_len, D]

        # Column vector of absolute positions t = 0..max_len-1, shape [max_len, 1].
        # Using float here so multiplication with frequency terms (also float) is clean.
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]

        # Build the geometric frequency terms: 10000^(-2i/D) for i in {0,1,2,..., D/2-1}.
        # We step by 2 because each frequency corresponds to a (sin, cos) channel pair.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [D/2] if D is even

        # Broadcast multiply: [max_len, 1] * [D/2] -> [max_len, D/2]
        # Fill even channels (0,2,4,...) with sines at those frequencies
        pe[:, 0::2] = torch.sin(position * div_term)

        # Fill odd channels (1,3,5,...) with cosines at the *same* frequencies
        # NOTE: If d_model is odd, use:
        #   pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe[:, 1::2] = torch.cos(position * div_term)

        # Store as a non-trainable buffer and add a leading batch-dim for easy broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings for the first T time steps and apply dropout.

        x: [B, T, D]
        returns: [B, T, D]
        """
        # Slice the first T positions and broadcast across the batch dimension.
        # self.pe: [1, max_len, D] -> [1, T, D]; x: [B, T, D]
        x = x + self.pe[:, : x.size(1)]

        # Regularize after the sum. Active in model.train(); no-op in model.eval().
        return self.dropout(x)

        # ---------------------------------------------------------------------------
        # Detailed line-by-line explanation & demo outputs (requested)
        # Demo config: d_model=8, max_len=6, dropout=0.2, batch B=2, length T=5
        #
        # Constructor (__init__)
        # 1) pe = torch.zeros(max_len, d_model)
        #    Allocates the table to hold all position vectors.
        #    Output (demo): shape [6, 8]; all zeros.
        #    [[0.,0.,0.,0.,0.,0.,0.,0.],
        #     ...
        #     [0.,0.,0.,0.,0.,0.,0.,0.]]
        #
        # 2) position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        #    Column vector of positions 0..max_len-1; unsqueeze(1) makes it [max_len,1] for broadcasting.
        #    Output: shape [6, 1] (shown transposed for brevity)
        #    [[0., 1., 2., 3., 4., 5.]]
        #
        # 3) div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #    Geometrically spaced angular frequencies for sin/cos pairs: 10000^{-2i/D}.
        #    Output: shape [4]
        #    [1.0000e+00, 1.0000e-01, 1.0000e-02, 1.0000e-03]
        #
        # 4) pe[:, 0::2] = torch.sin(position * div_term)
        #    Compute sines at those frequencies and write them into even channels (0,2,4,6).
        #    Broadcasting: [6,1] × [4] → [6,4].
        #    Output: shape [6, 4] (full matrix shown in the notebook cell).
        #
        # 5) pe[:, 1::2] = torch.cos(position * div_term)
        #    Cosines for odd channels (1,3,5,7) at the same frequencies as their even partner.
        #    Output: shape [6, 4] (full matrix shown above).
        #
        # 6) self.register_buffer('pe', pe.unsqueeze(0))
        #    Adds a batch dimension and stores as a buffer (tracked in state_dict, not trainable).
        #    Output: shape [1, 6, 8]; first 6 positions printed in the cell.
        #
        # Forward
        # Assume x.shape = [B, T, D] = [2, 5, 8].
        # 7) (implicit) x
        #    Your token embeddings pre-PE.
        #    Output example (sampled): shape [2, 5, 8]; first item printed in the cell.
        #
        # 8) x = x + self.pe[:, : x.size(1)]
        #    Slice T rows from positional table ([1, T, D]) and broadcast-add to [B, T, D].
        #    Output: shape [2, 5, 8]; first item:
        #    [[-1.5256,  0.2498, -0.6540, -0.6095, -0.1002,  0.3908, -0.9798, -0.6091],
        #     [ 0.1293,  0.8440, -0.6775,  0.7435, -0.2123,  2.6871,  0.2294,  1.4676],
        #     [ 0.2123, -1.5769,  0.8982,  1.1791,  0.8857,  1.2442, -0.6609,  1.8073],
        #     [ 1.2428, -1.1659, -1.9500, -0.4911,  0.0912,  0.3818, -0.7951,  0.8684],
        #     [ 1.1225, -0.7258,  0.5472,  0.1476,  0.2390,  1.0449,  0.1570,  0.5243]]
        #
        # 9) return self.dropout(x)
        #    Applies dropout (here p=0.2). In train mode, ~20% entries go to 0 and others are scaled by 1/(1-p)=1.25.
        #    Output (train): shape [2, 5, 8]; first item:
        #    [[-0.0000,  0.3122, -0.8175, -0.7619, -0.1252,  0.4885, -1.2247, -0.7614],
        #     [ 0.1617,  1.0550, -0.8469,  0.9294, -0.0000,  3.3588,  0.2868,  1.8345],
        #     [ 0.2654, -0.0000,  1.1228,  1.4739,  0.0000,  1.5553, -0.8261,  0.0000],
        #     [ 1.5535, -1.4574, -2.4375, -0.6139,  0.0000,  0.4773, -0.9938,  1.0855],
        #     [ 1.4032, -0.0000,  0.6840,  0.0000,  0.2988,  1.3061,  0.0000,  0.0000]]
        #    In eval mode, dropout is a no-op → returns exactly x + PE. Confirmed: Dropout(eval) equals input? -> True
        #
        # Quick mental model
        #  - position (time index) × div_term (frequency) → a grid of angles.
        #  - sin/cos of that grid → a bank of “positional waves” across channels.
        #  - Add those waves to embeddings so attention can sense position and relative offsets.
        #  - Dropout regularizes after the sum.
        #  sin/cos produce a phase code of position where shifts are rotations, similarity is a function of the offset, the scale is stable
        # ---------------------------------------------------------------------------
        # Worked example: positional encoding on 4 tokens (D=8, T=4, B=1, dropout=0)
        # --------------------------------------------------------------------------------
        # Goal: show exactly what `x = x + self.pe[:, : x.size(1)]` does on a tiny batch.
        #
        # Toy token embeddings BEFORE PE (given):
        #   t0 ("Alice"):  [ 0.20, -0.10,  0.30,  0.50, -0.40,  0.10,  0.20, -0.10]
        #   t1 ("eats"):   [ 0.00,  0.20, -0.10,  0.10,  0.30, -0.20,  0.10,  0.00]
        #   t2 ("two"):    [-0.20,  0.10,  0.10, -0.10,  0.10,  0.40, -0.10,  0.10]
        #   t3 ("apples"): [ 0.10,  0.10,  0.20, -0.20,  0.00,  0.10,  0.30, -0.10]
        #
        # 1) Frequencies used by sin/cos pairs
        #    For D=8, we have D/2 = 4 frequencies:
        #      div_term = 10000^{-2i/D}  for i=0..3  →  [1.0, 0.1, 0.01, 0.001]
        #
        # 2) Build the first T=4 rows of PE (even dims = sin, odd dims = cos, same frequency per pair)
        #    t = 0:
        #      sin(0·ω) = 0,  cos(0·ω) = 1
        #      PE[0] = [0.00000,  1.00000,   0.00000,  1.00000,   0.00000,  1.00000,   0.00000,  1.00000]
        #    t = 1:
        #      sin(1·[1,0.1,0.01,0.001]) = [0.84147, 0.09983, 0.01000, 0.00100]
        #      cos(1·[1,0.1,0.01,0.001]) = [0.54030, 0.99500, 0.99995, 0.9999995]
        #      PE[1] ≈ [0.84147,  0.54030,   0.09983,  0.99500,   0.01000,  0.99995,   0.00100,  0.9999995]
        #    t = 2:
        #      sin(2·[1,0.1,0.01,0.001]) = [0.90930, 0.19867, 0.02000, 0.00200]
        #      cos(2·[1,0.1,0.01,0.001]) = [-0.41615, 0.98007, 0.99980, 0.999998]
        #      PE[2] ≈ [0.90930, -0.41615,   0.19867,  0.98007,   0.02000,  0.99980,   0.00200,  0.9999980]
        #    t = 3:
        #      sin(3·[1,0.1,0.01,0.001]) = [0.14112, 0.29552, 0.03000, 0.00300]
        #      cos(3·[1,0.1,0.01,0.001]) = [-0.98999, 0.95534, 0.99955, 0.9999955]
        #      PE[3] ≈ [0.14112, -0.98999,   0.29552,  0.95534,   0.03000,  0.99955,   0.00300,  0.9999955]
        #
        # 3) Apply the forward rule:  x ← x + self.pe[:, :T]
        #    Shapes:
        #      x           : [B, T, D] = [1, 4, 8]
        #      self.pe[:,:T]: [1, 4, 8]  (broadcast along batch)
        #      result       : [1, 4, 8]
        #
        #    Element-wise sums per position (dropout=0 ⇒ no zeros, no rescaling):
        #
        #    Position 0 ("Alice"):   t0 + PE[0]
        #      [ 0.20, -0.10,  0.30,  0.50, -0.40,  0.10,  0.20, -0.10]
        #    + [ 0.00,  1.00,  0.00,  1.00,  0.00,  1.00,  0.00,  1.00]
        #    = [ 0.20,  0.90,  0.30,  1.50, -0.40,  1.10,  0.20,  0.90]
        #
        #    Position 1 ("eats"):    t1 + PE[1]
        #      [ 0.00,  0.20, -0.10,  0.10,  0.30, -0.20,  0.10,  0.00]
        #    + [ 0.84147,  0.54030,  0.09983,  0.99500,  0.01000,  0.99995,  0.00100,  0.9999995]
        #    ≈ [ 0.84147,  0.74030, -0.00017,  1.09500,  0.31000,  0.79995,  0.10100,  1.00000]
        #
        #    Position 2 ("two"):     t2 + PE[2]
        #      [-0.20,  0.10,  0.10, -0.10,  0.10,  0.40, -0.10,  0.10]
        #    + [ 0.90930, -0.41615,  0.19867,  0.98007,  0.02000,  0.99980,  0.00200,  0.9999980]
        #    ≈ [ 0.70930, -0.31615,  0.29867,  0.88007,  0.12000,  1.39980, -0.09800,  1.10000]
        #
        #    Position 3 ("apples"):  t3 + PE[3]
        #      [ 0.10,  0.10,  0.20, -0.20,  0.00,  0.10,  0.30, -0.10]
        #    + [ 0.14112, -0.98999,  0.29552,  0.95534,  0.03000,  0.99955,  0.00300,  0.9999955]
        #    ≈ [ 0.24112, -0.88999,  0.49552,  0.75534,  0.03000,  1.09955,  0.30300,  0.90000]
        #
        # Result of PositionalEncoding.forward (dropout off):
        #   [
        #     [ 0.20,  0.90,  0.30,  1.50, -0.40,  1.10,  0.20,  0.90],  # t=0 ("Alice")
        #     [ 0.84,  0.74, -0.00,  1.10,  0.31,  0.80,  0.10,  1.00],  # t=1 ("eats")
        #     [ 0.71, -0.32,  0.30,  0.88,  0.12,  1.40, -0.10,  1.10],  # t=2 ("two")
        #     [ 0.24, -0.89,  0.50,  0.76,  0.03,  1.10,  0.30,  0.90],  # t=3 ("apples")
        #   ]
        #
        # Why this helps (intuition):
        #   • The embedding encodes *what* the token is; the sinusoid encodes *where* it is.
        #   • Adding them gives each vector joint content+position information so attention can reason about both.
        # --------------------------------------------------------------------------------

# -------------------------
# Scaled Dot-Product Attention
# -------------------------
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (per head)
    --------------------------------------
    Purpose
      Compute attention as a content-based weighted average of values V, where weights come from
      softmax(QK^T / sqrt(d_k)) after applying an optional mask.

    Shapes
      Q: [B, H, T_q, d_k]  – queries per head
      K: [B, H, T_k, d_k]  – keys per head
      V: [B, H, T_k, d_v]  – values per head (often d_v=d_k)
      attn_mask: broadcastable to [B, H, T_q, T_k] (True = keep, False = mask)

    Returns
      out:   [B, H, T_q, d_v]  – attended representations per head
      attn:  [B, H, T_q, T_k]  – attention probabilities (useful for visualization)

    Notes
      • The 1/sqrt(d_k) scale stabilizes logits before softmax.
      • Masking with -inf ensures masked positions receive probability 0 after softmax.
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout on attention weights (regularization)

    def forward(
        self,
        Q: torch.Tensor,  # [B,H,T_q,d_k]
        K: torch.Tensor,  # [B,H,T_k,d_k]
        V: torch.Tensor,  # [B,H,T_k,d_v]
        attn_mask: Optional[torch.Tensor] = None,  # bool mask broadcastable to [B,H,T_q,T_k]; True=keep, False=mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Q: [B, H, T_q, d_k] - queries for each head (B=batch, H=heads, T_q=query positions, d_k=per-head dim)
        # K: [B, H, T_k, d_k] - keys for each head (T_k=key positions)
        # V: [B, H, T_k, d_v] - values for each head (d_v usually == d_k)
        # attn_mask: [B, H, T_q, T_k] or broadcastable; True=keep, False=mask. Used to mask out pads or future tokens.

        # Extract the per-head dimension (d_k) from Q's last dimension.
        # This is needed for scaling the dot products to stabilize gradients.
        # For example, if Q.shape = [32, 8, 20, 64], then d_k = 64.
        d_k = Q.size(-1)  # Per-head key/query width for scaling (scalar)

        # Compute raw attention scores (compatibility) between queries and keys.
        # Q: [B,H,T_q,d_k], K: [B,H,T_k,d_k]
        # K.transpose(-2, -1): [B,H,d_k,T_k]
        # torch.matmul(Q, K^T): [B,H,T_q,d_k] x [B,H,d_k,T_k] -> [B,H,T_q,T_k]
        # Each score[i,j,:,:] is the dot product between each query (at T_q) and each key (at T_k) for head j in batch i.
        # We divide by sqrt(d_k) to prevent large dot products, which would push softmax into regions with very small gradients.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,T_q,T_k]
        # Intuition: each score[i, h, tq, tk] is a dot-product similarity between query at tq and key at tk
        # (after learned projections). Larger → more relevant. Division by sqrt(d_k) keeps logits in a good
        # range so softmax doesn’t saturate when d_k is large.

        # If a mask is provided, set masked positions to -inf so their softmax becomes 0.
        # attn_mask: [B,H,T_q,T_k], with True=keep, False=mask.
        # ~attn_mask inverts mask: True for masked positions.
        # masked_fill broadcasts mask to match scores shape.
        # This is used for padding (so model doesn't attend to pad tokens) and for causal masking in the decoder.
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float('-inf'))

        # Apply softmax along the last dimension (over keys) to get attention weights.
        # For each query position, the weights sum to 1 over all key positions.
        # This produces a probability distribution over keys for each query (for each head, each batch).
        # attn: [B,H,T_q,T_k]
        attn = torch.softmax(scores, dim=-1)
        # Now, for each query position, attn sums to 1 across all key positions (a probability distribution).

        # Apply dropout to the attention weights (regularizes which keys are attended to).
        # During training, randomly zeros some attention weights and rescales others.
        # During eval, dropout is a no-op (all weights kept).
        attn = self.dropout(attn)

        # Compute the weighted sum of values according to the attention weights.
        # attn: [B,H,T_q,T_k], V: [B,H,T_k,d_v]
        # torch.matmul(attn, V): [B,H,T_q,T_k] x [B,H,T_k,d_v] -> [B,H,T_q,d_v]
        # For each query, this is a weighted average of the value vectors, using attn as weights.
        out = torch.matmul(attn, V)  # [B,H,T_q,d_v]

        # Return:
        # out: [B,H,T_q,d_v] - the attended representations for each query position, head, and batch.
        # attn: [B,H,T_q,T_k] - the attention weights (probabilities) used for each query over all keys (can be visualized).
        return out, attn


# -------------------------
# Multi-Head Attention
# -------------------------
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA)
    --------------------------
    Purpose
      • Run H independent scaled dot-product attentions in parallel on smaller subspaces, then concatenate and mix them. Different heads can focus on different positions/relations.

    Shapes
      Inputs
        q: [B, T_q, D]  – queries
        k: [B, T_k, D]  – keys
        v: [B, T_k, D]  – values
        attn_mask: broadcastable to [B, H, T_q, T_k] (True = keep, False = mask)
      Internals
        After projection & split: Q,K,V → [B, H, T_*, d_k] with d_k = D/H
      Output
        out: [B, T_q, D]

    Math (per head h)
      attn_h = softmax( (Q_h · K_h^T) / sqrt(d_k)  + mask ) · V_h
      out   = W_o · concat_h( attn_h )

    Notes
      • Scaling by 1/sqrt(d_k) stabilizes softmax magnitudes.
      • Masks zero out disallowed attention (pads/causal) by setting scores to -inf before softmax.
    """
    # --- Single-Head vs Multi-Head attention (for one query position t) ---
    # SHA: one attention distribution α over all keys → one weighted average
    #      z = Σ_s α_s · V(s). All evidence (e.g., "two" and "apples") is mixed once.
    # MHA: H independent attention distributions α^(1..H) with separate projections
    #      (W_q^(h), W_k^(h), W_v^(h)). Each head produces its own z^(h) that can
    #      focus on different aspects (syntax vs identity, long-range vs local).
    #      Concat [z^(1); … ; z^(H)] and mix with W_o.
    # Benefit: head-wise separability before fusion; signals don’t have to compromise
    #          in a single softmax. Both SHA and MHA see all tokens (subject to masks);
    #          the advantage is separability, not larger context.
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, 'd_model must be divisible by number of heads'  # Ensure even split across heads
        self.d_model = d_model  # Model width D
        self.h = h              # Number of heads H
        self.d_k = d_model // h # Per-head width d_k = D/H
        # --- What is a "head" and why split D into H heads? ---
        # A *head* is one parallel attention unit. Instead of attending once in the full D-dim space,
        # we run H attentions in smaller subspaces of width d_k = D/H, then concatenate results.
        # Benefits:
        #  • Diversity: different heads can specialize in different relations (syntax, coreference, long-range vs local).
        #  • Efficiency: overall compute stays ~O(B * T^2 * D) while getting H different softmax patterns.
        #  • Stability: scaling by sqrt(d_k) keeps attention logits well-behaved per head.

        # Learned projections to Q, K, V spaces and final output mix (all keep width D)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        # Why project with W_q, W_k, W_v instead of using raw q/k/v?
        # We learn *what to compare* and *what to retrieve*:
        #   Q = q W_q,   K = k W_k,   V = v W_v
        # Scores become a learned bilinear similarity: (q W_q) (k W_k)^T = q (W_q W_k^T) k^T
        # • Without these, attention would be plain dot products on raw features (less expressive).
        # • Separate W_q and W_k allow asymmetric comparisons (queries vs keys need not share coordinates).
        # • W_v selects which information to carry forward in the weighted sum (not just averaging raw inputs).
        # • In cross-attn, these projections map encoder/decoder spaces into a common head width d_k.

        self.attn = ScaledDotProductAttention(dropout)  # Core per-head attention: softmax(QK^T/√d_k)·V
        self.dropout = nn.Dropout(dropout)              # Dropout on the MHA output (regularization)

        self._reset_parameters()  # Xavier-init for stable training

    def _reset_parameters(self):
        # Xavier/Glorot uniform: weights ~ U(-a, a), with a = sqrt(6 / (fan_in + fan_out)).
        # For Linear(D, D): fan_in = fan_out = D → a = sqrt(3 / D).
        # This keeps activation/gradient variance roughly constant across layers at init time.
        for m in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(m.weight)  # Variance-preserving init for linear layers
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape                              # x: [B,T,D]
        x = x.view(B, T, self.h, self.d_k).permute(0, 2, 1, 3)  # → [B,T,H,d_k] → [B,H,T,d_k]
        # NOTE: view() requires contiguous memory (Linear outputs are contiguous).
        # If you refactor and see a stride error, call .contiguous() before view(), or use .reshape(...).
        return x  # [B,H,T,d_k]

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, d = x.shape                           # x: [B,H,T,d_k]
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, H * d)  # → [B,T,H,d_k] → [B,T,D]
        # Why merge back to [B,T,D]?
        # • Residual add expects the same width D as the input stream.
        # • Downstream LayerNorm/FFN blocks operate on width D.
        # • W_o (next line in forward) learns how to mix the concatenated heads.
        return x  # [B,T,D]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1) Project to Q/K/V, staying at width D
        Q = self._split(self.W_q(q))  # [B,H,T_q,d_k]
        K = self._split(self.W_k(k))  # [B,H,T_k,d_k]
        V = self._split(self.W_v(v))  # [B,H,T_k,d_k]

        # 2) Scaled dot-product attention per head (mask broadcastable to [B,H,T_q,T_k])
        out, _ = self.attn(Q, K, V, attn_mask)  # out: [B,H,T_q,d_k]

        # 3) Concatenate heads and mix with W_o
        out = self._merge(out)   # [B,T_q,D]
        # Final width is D because we concatenated H heads (H * d_k = D). We return one vector per query position T_q.
        out = self.W_o(out)      # [B,T_q,D]
        # W_o learns to fuse information from different heads into the model space; dropout regularizes before residual.
        return self.dropout(out) # Dropout before residual/add-norm in the surrounding block


# -------------------------
# Position-wise Feed-Forward (ReLU)
# -------------------------
class PositionwiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN)
    ----------------------------------------
    Purpose
      • Apply the same 2-layer MLP independently at every time step (like a 1x1 conv on channels).
      • Pattern: expand features, apply nonlinearity + dropout, project back to model width.

    Shapes
      Input  x : [B, T, D]  (PyTorch Linear accepts any leading dims; treat [..., D])
      Hidden   : [B, T, d_ff]
      Output   : [B, T, D]

    Math (per position t)
      h1 = ReLU( x_t W1 + b1 )            with W1 ∈ ℝ^{D×d_ff}, b1 ∈ ℝ^{d_ff}
      y_t = ( Dropout(h1) ) W2 + b2       with W2 ∈ ℝ^{d_ff×D}, b2 ∈ ℝ^{D}

    Notes
      • Typically d_ff ≈ 4·D (e.g., 2048 when D=512) to increase capacity between attention blocks.
      • This module transforms features *within* a position; attention handles mixing *across* positions.

    Why expand (D → d_ff)?
      • More degrees of freedom for the activation: a wider hidden layer gives ReLU/GELU more room to form rich combinations.
      • Richer per-token recomposition: attention already mixed across positions; expansion lets us recombine *within* a token.
      • Capacity without changing the external width: go wide inside the block, then return to D so residuals and later layers match.
      • Important: If there were *no* nonlinearity after expansion, `lin2·lin1` would collapse to a single linear map—expansion would be pointless.

    Why a nonlinearity (ReLU)?
      • It’s *nonlinear*: ReLU(z)=max(0,z) breaks linearity (so stacked linears don’t collapse into one).
      • Formal check: a function f is linear if f(a·x1 + b·x2) = a·f(x1) + b·f(x2) for all a,b,x1,x2 (and affine if f(x)=Ax+b). ReLU(z)=max(0,z) violates this: ReLU(-1+1)=0 ≠ ReLU(-1)+ReLU(1)=1. Therefore ReLU is a nonlinear operation.
      • Gating & sparsity: negatives go to 0 → input-dependent switching; different subsets of hidden units fire for different tokens.
      • Complements attention: softmax is a nonlinearity over *positions*; ReLU is a nonlinearity over *channels* within each token.

    If you remove the FFN
      • Layer becomes "attention-only": mostly a contextual weighted average plus linear mixing—much weaker per-token transformation.
      • Expect worse convergence/quality; you can shrink `d_ff` to save params, but removing FFN hurts most tasks.
    """
    # --- Tiny worked example: "Alice eats two apples" (B=1, T=6) ---
    # After attention + Add&Norm, each token vector x_t ∈ ℝ^D already carries context.
    #   e.g., for the token "apples", x_apples includes:
    #     • noun identity features
    #     • a plural/count signal pulled from "two" by attention
    #     • local syntax cues (object of "eats")
    # The FFN then runs independently per position:
    #   h1 = ReLU(W1 · x_t + b1) activates specialized detectors, e.g.:
    #     – "is noun & count>1?" → likely fires on "apples"
    #     – "edible object after 'eat'?" → may also fire
    #   y_t = W2 · Dropout(h1) projects back to width D for the residual stream.
    # Shapes (paper defaults D=512, d_ff=2048): [1,6,512] → [1,6,2048] → [1,6,2048] → [1,6,512].
    # Why expand to d_ff then project back?
    #   Capacity for rich nonlinear per-token transforms, while returning to D so residual Add&Norm works.
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # First affine: expands channel width from D → d_ff so ReLU can form richer combinations.
        # Applies independently to every time step / batch item.
        self.lin1 = nn.Linear(d_model, d_ff)     # W1: [D, d_ff], b1: [d_ff]; input [..., D] → [..., d_ff]
        # Expand features D→d_ff (e.g., 512→2048). This increases capacity so the following
        # nonlinearity can build rich, selective combinations per token. Without the activation,
        # expansion wouldn’t help because two Linear layers would collapse into one.

        # Second affine: projects back to model width so shapes match the residual stream.
        self.lin2 = nn.Linear(d_ff, d_model)     # W2: [d_ff, D], b2: [D];   [..., d_ff] → [..., D]
        # Project back to model width D so residual/add-norm and subsequent layers see the standard shape [B,T,D].

        # Dropout applied between ReLU and the final projection (regularization).
        self.dropout = nn.Dropout(dropout)

        # Initialize weights/biases for stable early training.
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier/Glorot uniform initialization:
        #   weights ~ U(-a, a) with a = sqrt(6 / (fan_in + fan_out)).
        # For Linear(in=D, out=d_ff) and Linear(in=d_ff, out=D), this keeps activation/gradient
        # variances roughly constant across layers at initialization time.
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.constant_(self.lin1.bias, 0.0)
        nn.init.constant_(self.lin2.bias, 0.0)
        # Note: A common variant uses Kaiming/He init for lin1 when followed by ReLU, and Xavier for lin2.
        # Your Xavier–Xavier choice is standard and works well in practice.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (applied position-wise)
          x: [B, T, D]  →  y: [B, T, D]

        Step-by-step (implicit in the one-liner below):
          1) h1 = lin1(x)         # [B,T,D]   → [B,T,d_ff]   (WHY: widen to give the activation more room/capacity)
          2) h1 = ReLU(h1)        # [B,T,d_ff]              (WHY: nonlinearity; prevents collapse of two linears; gating/sparsity)
          3) h1 = Dropout(h1)     # [B,T,d_ff] (train-time regularization; no-op in eval)
          4) y  = lin2(h1)        # [B,T,d_ff] → [B,T,D] (back to model width for residual)
        """
        # Example at token "apples":
        #   • Attention has already brought in quantity info from "two".
        #   • ReLU(W1 · x_apples + b1) turns on plural-sensitive units (negatives → 0).
        #   • W2 then reinforces features that help downstream prediction prefer a plural form.
        #   • This happens at every position independently (no time mixing inside FFN).
        # Keep concise compute while documenting the full logic above:
        # Summary: expand (D→d_ff) → ReLU (nonlinear gating) → Dropout (regularize) → project back (d_ff→D).
        # This supplies the per-token nonlinear transformation that attention alone cannot provide.
        return self.lin2(self.dropout(F.relu(self.lin1(x))))


# -------------------------
# Sublayer (Residual + Dropout + LayerNorm) — Post-LN
# -------------------------
class SublayerConnection(nn.Module):
    """
    SublayerConnection (Add & Norm, post-LayerNorm)
    ------------------------------------------------
    Purpose
      • Wrap a sublayer (e.g., MHA or FFN) with Dropout → Residual Add → LayerNorm, as in the original Transformer.

    Shapes
      Input  x            : [B, T, D]  – residual stream (batch, time, model width)
      Input  sublayer_out : [B, T, D]  – output produced by the inner sublayer on the same shape
      Output y            : [B, T, D]

    Math (per token t)
      y_t = LN( x_t + Dropout( sublayer_out_t ) )

    Why this block
      • Dropout: regularize the sublayer’s contribution.
        Why: regularizes the *update* from the sublayer so the model doesn’t over‑rely on any one head/feature pathway; encourages redundancy.
        If you remove it: overfitting rises (train–val gap widens), heads/neurons co‑adapt (one path does all the work), attention can become too peaky/brittle.
        Symptoms: larger train–val gap, fragile predictions, poorer robustness to shift.

      • Residual add: provides an identity path for gradients and lets the sublayer learn a *correction* to x.
        Why: creates an identity route for signals/gradients → deep stacks train; the sublayer learns Δ to add to x, not a full rewrite (easier optimization).
        If you remove it: vanishing/exploding gradients, unstable/slow training, shallow useful depth; the model cannot “do nothing” locally.
        Symptoms: noisy/unstable loss, gradients ~0 or huge, high sensitivity to init/LR.

      • LayerNorm: stabilizes activation statistics per token (normalize over the D channels), improving training stability.
        Why: keeps post‑residual activations well‑scaled per token; reduces covariate shift; independent of batch size/sequence length (unlike BatchNorm).
        If you remove it: activation scales drift across depth → saturation/NaNs/Inf, erratic gradients; training becomes highly sensitive to hyperparams.
        Symptoms: loss spikes, intermittent NaNs, attention/FFN activations with extreme magnitudes, need for very careful LR schedules.

    Post-LN note
      • This is *post*-LayerNorm (Add → Norm), matching the original paper. Many modern variants use *pre*-LayerNorm (Norm → Sublayer → Add)
        for improved training at very deep scales; your implementation intentionally follows post-LN.

    Why this formula works well (deeper rationale)
      • We compute:  y = LayerNorm( x + Dropout( f(x) ) ), where f is MHA or FFN.

      1) Residual add (identity path ⇒ easy optimization)
         • Let z = x + Dropout(f(x)). The Jacobian ∂z/∂x ≈ I + J_{Dropout∘f}(x). The identity term gives a direct route for signals and
           gradients, preventing vanishing/exploding through depth and letting the sublayer learn a *correction* Δ(x) instead of a full rewrite.

      2) Dropout on the update (regularization without destroying the signal)
         • Inverted dropout: Dropout(u) = M ⊙ u / (1-p) with M_d ~ Bernoulli(1-p) ⇒ E[Dropout(f(x))] = f(x).
           Only the *correction* path is noisy; the identity path x is preserved. This reduces co‑adaptation/overfitting and yields more robust patterns.

      3) LayerNorm (stable scales for smooth training)
         • Per-token normalization across D channels: LN(z)_d = γ_d (z_d − μ)/sqrt(σ²+ε) + β_d.
           Keeps activations in a numerically friendly range for softmax/ReLU, reduces covariate shift, and its bounded Jacobian stabilizes gradients.

      4) The combo (synergy)
         • Residual fixes depth/gradient issues; Dropout combats overfitting on the update; LayerNorm controls activation scale.
           Together they preserve shape [B,T,D], keep the residual stream coherent, and consistently improve convergence and validation quality.

      5) Ordering (post‑LN vs pre‑LN)
         • This implementation is post‑LN (Add → Norm), faithful to the original Transformer. Pre‑LN (Norm → Sublayer → Add) often helps at extreme depth,
           but the above benefits hold either way.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # LayerNorm over the feature dimension D; applied independently at each (batch, time) position.
        self.norm = nn.LayerNorm(d_model)
        # Dropout to regularize the sublayer output before adding it back to the residual stream.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x           : [B, T, D] residual/input stream
          sublayer_out: [B, T, D] result from an inner sublayer (e.g., MHA(x, ...), FFN(x))
        Returns:
          y           : [B, T, D] normalized residual sum
        """
        # 1) Regularize the sublayer output (train-mode only; no-op in eval).
        #    Shape stays [B, T, D].
        dropped = self.dropout(sublayer_out)  # [B, T, D]

        # 2) Residual addition: identity path + corrective update from the sublayer.
        #    Adds elementwise over the last dimension; shape preserved: [B, T, D].
        x = x + dropped  # [B, T, D]

        # 3) Normalize per token across channels (post-LN). Keeps shape [B, T, D].
        return self.norm(x)  # [B, T, D]


# -------------------------
# Encoder & Decoder Layers (Post-LN as in the original paper)
# -------------------------
class EncoderLayer(nn.Module):
    """
    EncoderLayer (Self-Attn → FFN), each wrapped with Add & Norm (post-LN)
    ----------------------------------------------------------------------
    Purpose
      • One encoder block performs two complementary operations:
        1) **Self-Attention (MHA)** to mix information **across positions** (context gathering).
        2) **Position-wise FFN** to nonlinearly transform **within each position** (feature recomposition).
      • Each sublayer is wrapped by **Dropout → Residual Add → LayerNorm** (your SublayerConnection).

    Shapes
      Input  x      : [B, S, D]  (batch, source length S, model width D)
      Mask   src_mask: broadcastable to [B, H, S, S] (True = keep, False = mask pads)
      Output y      : [B, S, D]

    Why **two** sublayers?
      • **Self-Attention** answers “*where should this token look?*”
        – Computes content-based weights over all positions: softmax(QK^T/√d_k)·V.
        – Learns long-range dependencies, coreference, syntax, etc. (cross-token mixing).
        – Limitation alone: mostly a (softmax-weighted) **linear** mixture of values once the weights are set.
      • **FFN** answers “*what should we do with what was read?*”
        – A 2-layer MLP (expand → nonlinearity → project) applied per position.
        – Adds strong **nonlinearity** and capacity to re-encode the attended features (within-token processing).
        – Limitation alone: no cross-token communication.
      • **Together**: expressive + contextual. Attention handles **relations**; FFN provides **nonlinear feature recomposition**.

    What breaks if you drop one?
      • Drop FFN → “attention-only”: becomes largely a contextual weighted average + linear mixing → lower quality on most tasks.
      • Drop attention → only FFN: no cross-token context; each token transforms itself in isolation.
      • Single Add&Norm for both instead of two: less stable optimization; you lose the normalization/regularization checkpoint between the very different ops.

    Flow (post-LN, per block)
      1) attn_out = MHA(x, x, x, src_mask)       # self-attn across S
      2) x ← Add&Norm(x, attn_out)               # Dropout → Residual Add → LayerNorm
      3) ffn_out = FFN(x)                        # per-position nonlinearity
      4) x ← Add&Norm(x, ffn_out)                # Dropout → Residual Add → LayerNorm
    """
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention: mixes information across positions S using H heads in subspaces of size d_k = D/H.
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        # Position-wise 2-layer MLP: expand D→d_ff, apply ReLU + Dropout, project back d_ff→D.
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        # Add & Norm wrappers (post-LN) for each sublayer output.
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args
          x        : [B, S, D]
          src_mask : broadcastable to [B, H, S, S] (True=keep, False=mask)
        Returns
          y        : [B, S, D]
        """
        # ---------------------------------------------------------------------
        # Tiny intuition on self-attention (toy sentence, B=1, S=6)
        #   "<bos> Alice eats two apples <eos>"
        # Heads can learn complementary patterns, e.g.:
        #   • Head A (syntax/role):   "eats" attends to subject "Alice" and object "apples"
        #       weights_q=eats → [bos, Alice, eats, two, apples, eos]
        #                          [0.05, 0.40, 0.05, 0.05, 0.40, 0.05]
        #   • Head B (local):         "apples" attends to nearby quantifier "two"
        #       weights_q=apples → [bos, Alice, eats, two, apples, eos]
        #                            [0.05, 0.05, 0.20, 0.60, 0.05, 0.05]
        # Each row softmaxes to 1 and is used to weight-sum the V vectors.
        # The concatenated head outputs are linearly mixed by W_o, then wrapped
        # by residual+LayerNorm below.
        # Shapes in this call:
        #   x: [B,S,D], attn_out: [B,S,D] (contextualized representations, same shape)
        # ---------------------------------------------------------------------
        # 1) Self-attention over the source sequence: q=k=v=x for self-attn.
        #    Produces a context-enriched representation per position. Shape: [B, S, D].
        attn_out = self.self_attn(x, x, x, src_mask)  # [B, S, D]

        # 2) Wrap with Dropout → Residual Add → LayerNorm (post-LN). Shape preserved.
        x = self.sublayer1(x, attn_out)  # [B, S, D]

        # 3) Per-position nonlinearity: expand → ReLU → Dropout → project back. Shape: [B, S, D].
        ffn_out = self.ffn(x)  # [B, S, D]

        # 4) Second Add & Norm wrapper after FFN. Shape preserved.
        x = self.sublayer2(x, ffn_out)  # [B, S, D]

        return x  # [B, S, D]


class DecoderLayer(nn.Module):
    """
    DecoderLayer (Masked Self-Attn → Cross-Attn → FFN), each wrapped with Add & Norm (post-LN)
    -----------------------------------------------------------------------------------------
    High-level I/O: [B, T, D] → [B, T, D]
      • self_attn : masked (causal) self-attention over the decoder tokens (q=k=v=x)
      • cross_attn: encoder–decoder attention (q=x, k=v=enc_out)
      • ffn       : position-wise 2-layer MLP with ReLU and dropout (D→d_ff→D)
      • sublayer* : wrappers that do Dropout → Residual Add → LayerNorm (post-LN)
    """
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Multi-Head masked self-attention on the decoder stream (queries/keys/values all from x)
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        # Multi-Head cross-attention: queries come from decoder stream x; keys/values come from the encoder output
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        # Position-wise Feed-Forward Network: D → d_ff (ReLU + Dropout) → D, applied independently at each time step
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        # Post-LN Add&Norm wrappers: wrap each sublayer output with Dropout → Residual Add → LayerNorm
        self.sublayer1 = SublayerConnection(d_model, dropout)  # wraps masked self-attention
        self.sublayer2 = SublayerConnection(d_model, dropout)  # wraps cross-attention
        self.sublayer3 = SublayerConnection(d_model, dropout)  # wraps FFN

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args
          x       : [B, T, D] decoder stream (embeddings+positions after previous decoder layer)
          enc_out : [B, S, D] encoder memory (context to attend to)
          tgt_mask: [B, 1, T, T] bool keep-mask for decoder self-attn (padding + causal); broadcasts to [B, H, T, T]
          src_mask: [B, 1, 1, S] bool keep-mask for cross-attn over encoder (padding only); broadcasts to [B, H, T, S]
        Returns
          y       : [B, T, D] same shape as input x
        """
        # ---------------------------------------------------------------------
        # How decoding uses masks & source context (toy view)
        # Assume at time step t we have generated a prefix on the decoder:
        #   y_prefix = ["<bos>", "Alice", "eats", ...]   (length t)
        # Masked self-attn:
        #   • Query at position t can only see positions ≤ t (lower-triangular causal mask).
        # Cross-attn:
        #   • The decoder representation at t queries the encoder memory enc_out [B,S,D].
        #   • For our example, when producing "apples", the query often attends strongly
        #     to encoder tokens "two" and "apples" to retrieve quantity/object semantics.
        # Resulting shapes in this block:
        #   self_out:  [B,T,D]  (masked self-attn output)
        #   cross_out: [B,T,D]  (alignment-informed features from the source)
        #   ffn_out:   [B,T,D]  (nonlinear per-position recomposition)
        # ---------------------------------------------------------------------
        # 1) Masked self-attention over the decoder tokens (no peeking ahead).
        #    Internals per head: Q,K,V ∈ [B,H,T,d_k]; scores ∈ [B,H,T,T], masked by tgt_mask, softmax → weighted V.
        self_out = self.self_attn(x, x, x, tgt_mask)          # [B, T, D]

        # 2) Add & Norm (post-LN) around masked self-attn: y = LN(x + Dropout(self_out))
        x = self.sublayer1(x, self_out)                       # [B, T, D]

        # 3) Cross-attention: decoder queries attend to encoder memory (alignment/retrieval from source tokens).
        #    q = x ∈ [B,T,D], k = v = enc_out ∈ [B,S,D]; scores ∈ [B,H,T,S], masked by src_mask on encoder pads.
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)  # [B, T, D]

        # 4) Add & Norm (post-LN) around cross-attn
        x = self.sublayer2(x, cross_out)                      # [B, T, D]

        # 5) Position-wise FFN (applied independently at each time step): expand → ReLU → Dropout → project back.
        ffn_out = self.ffn(x)                                  # [B, T, D]

        # 6) Add & Norm (post-LN) around FFN
        x = self.sublayer3(x, ffn_out)                        # [B, T, D]

        # 7) Output of this decoder layer (fed to the next decoder layer or generator)
        return x


# -------------------------
# Encoder / Decoder stacks
# -------------------------
class Encoder(nn.Module):
    """
    Encoder stack
    -------------
    Purpose
      • Turn a source token sequence into a contextual representation using:
        (a) token embeddings + sinusoidal positional encodings, then
        (b) N repeated blocks of Self-Attention → FFN (each wrapped by Dropout → Residual Add → LayerNorm).

    Shapes
      vocab        : size of the source vocabulary (int)
      d_model (D)  : model/embedding width (int)
      N            : number of identical encoder layers (int)
      h            : number of attention heads per layer (int)
      d_ff         : hidden width of the FFN sublayer (int)
      dropout      : dropout probability (float)

      Forward I/O
        src      : [B, S]        token IDs (batch B, source length S)
        src_mask : [B, 1, 1, S]  or broadcastable (True=keep, False=mask)
        return   : [B, S, D]     contextualized encoder representations
    """
    def __init__(self, vocab: int, d_model: int, N: int, h: int, d_ff: int, dropout: float):
        super().__init__()  # Initialize nn.Module internals (state dict, device transfer, etc.)

        # Token embedding table: maps token IDs → dense vectors in ℝ^D.
        # Input IDs of shape [B, S] become embeddings of shape [B, S, D] in forward().
        self.embed = nn.Embedding(vocab, d_model)

        # Fixed sinusoidal positional encoding: produces a [1, max_len, D] table
        # and adds the first S rows to the token embeddings so attention can sense order.
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        # The encoder is a *stack* of N identical layers.
        # Each EncoderLayer does: Self-Attn (across positions) → FFN (within position),
        # with Add&Norm after each sublayer (post-LN).
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])

        # Initialize parameters that benefit from a sane starting distribution.
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize embedding weights with a small normal distribution.
        # Reason: avoids large initial logits/activations and helps early optimization.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args
          src      : [B, S]        token IDs (ints)
          src_mask : [B, 1, 1, S]  broadcastable keep-mask (True=keep, False=mask pads)
        Returns
          x        : [B, S, D]     encoder output (to be consumed by the decoder cross-attn)

        Step-by-step
          1) Embedding lookup  : IDs → vectors                      [B,S]   → [B,S,D]
          2) Scale by √D       : stabilize magnitude of embeddings  [B,S,D] → [B,S,D]
          3) Add positions (+ dropout inside PE)                    [B,S,D] → [B,S,D]
          4) Pass through N EncoderLayers (self-attn + FFN)         [B,S,D] → [B,S,D]
        """
        # 1) IDs → embeddings: look up each token id in the table.
        #    x shape: [B, S, D]
        x = self.embed(src)  # [B,S,D]
        # ---------------------------------------------------------------------
        # Demo: what the embedding lookup returns (illustrative numbers only)
        # We map token ids to vectors by *gathering rows* from the embedding table.
        # Example tiny vocab and ids:
        #   0:<pad>  1:<unk>  2:<bos>  3:<eos>  4:Alice  5:eats  6:two  7:apples
        # Suppose d_model = 4 and the embedding matrix E has these rows for the tokens we use:
        #   E[2] (<bos>)   = [ 0.50, -0.20,  0.10,  0.30]
        #   E[4] (Alice)   = [ 0.20, -0.10,  0.30,  0.50]
        #   E[5] (eats)    = [ 0.00,  0.20, -0.10,  0.10]
        #   E[6] (two)     = [-0.20,  0.10,  0.10, -0.10]
        #   E[7] (apples)  = [ 0.10,  0.10,  0.20, -0.20]
        #   E[3] (<eos>)   = [ 0.40,  0.05, -0.15,  0.00]
        #
        # For input ids (B=1, S=6): [2, 4, 5, 6, 7, 3]  # "<bos> Alice eats two apples <eos>"
        # The embedding lookup produces x with shape [1, 6, 4]:
        # x[0] =
        # [
        #   [ 0.50, -0.20,  0.10,  0.30],   # <bos>    (E[2])
        #   [ 0.20, -0.10,  0.30,  0.50],   # Alice    (E[4])
        #   [ 0.00,  0.20, -0.10,  0.10],   # eats     (E[5])
        #   [-0.20,  0.10,  0.10, -0.10],   # two      (E[6])
        #   [ 0.10,  0.10,  0.20, -0.20],   # apples   (E[7])
        #   [ 0.40,  0.05, -0.15,  0.00],   # <eos>    (E[3])
        # ]
        # (Numbers above are illustrative to show the *shape* and *lookup* behavior only.)
        # ---------------------------------------------------------------------

        # 2) Scale embeddings by √D (as in the original paper) so that positional encodings
        #    and projected features have comparable magnitudes at initialization.
        x = x * math.sqrt(self.embed.embedding_dim)  # [B,S,D], same shape; numeric scale changes only

        # 3) Add sinusoidal positional encodings (and apply dropout inside PositionalEncoding).
        #    Shape preserved: [B, S, D]
        x = self.pos(x)  # [B,S,D]

        # 4) Run the stack of N encoder layers. Each layer:
        #      - mixes information across positions via self-attention (uses src_mask to ignore pads)
        #      - applies a position-wise FFN with nonlinearity
        #      - wraps each sublayer in Dropout → Residual Add → LayerNorm (post-LN)
        for layer in self.layers:
            # layer(x, src_mask) keeps shape [B, S, D]
            x = layer(x, src_mask)  # [B,S,D]

        # 5) Return the contextualized source representations, same shape as after embeddings.
        return x  # [B,S,D]


class Decoder(nn.Module):
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
        # Conceptual note (embedding lookup *before* scaling and positions):
        # Given the same toy vocab and sentence as in the encoder example:
        #   ids (B=1, T=6): [2, 4, 5, 6, 7, 3]
        # The call `self.embed(tgt)` alone would gather the same rows from E
        # and produce a tensor of shape [1, 6, d_model], e.g.:
        # [
        #   [ 0.50, -0.20,  0.10,  0.30],
        #   [ 0.20, -0.10,  0.30,  0.50],
        #   [ 0.00,  0.20, -0.10,  0.10],
        #   [-0.20,  0.10,  0.10, -0.10],
        #   [ 0.10,  0.10,  0.20, -0.20],
        #   [ 0.40,  0.05, -0.15,  0.00],
        # ]
        # We do not duplicate the scaling-by-√D or positional encoding comments here, per request.
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x


# -------------------------
# Full Transformer
# -------------------------
class Transformer(nn.Module):
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
        src: torch.Tensor,   # [B,S]
        tgt_in: torch.Tensor,  # [B,T]
        src_mask: Optional[torch.Tensor],  # [B,1,1,S]
        tgt_mask: Optional[torch.Tensor],  # [B,1,T,T]
    ) -> torch.Tensor:
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt_in, enc_out, tgt_mask, src_mask)
        # The generator is a linear layer W∈ℝ^{D×V} that maps each decoder vector
        # at each time step to unnormalized logits over the target vocab:
        #   logits = dec_out @ W + b   → shape [B, T, V]
        # Training:   apply CrossEntropyLoss over these logits vs. target ids.
        # Inference:  apply softmax to get probabilities; pick next token by
        #             greedy/beam sampling. The model is run autoregressively
        #             with the causal mask so each step only sees the prefix.
        return self.generator(dec_out)


# -------------------------
# Masks (bool True = keep, False = mask)
# -------------------------

def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B,1,1,S] mask for encoder & cross-attention."""
    # True = keep (valid token), False = mask (pad). Broadcasts to [B,H,T_q,S] in cross-attn.
    return (src != pad_id).unsqueeze(1).unsqueeze(1)


def make_subsequent_mask(T: int, device=None) -> torch.Tensor:
    """[1,1,T,T] lower-triangular keep-mask for decoder self-attention."""
    # Lower-triangular True (≤ diagonal) enforces causality: position t cannot attend to >t.
    m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return m.unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[B,1,T,T] = padding mask AND causal mask."""
    # Combines padding mask over T with causal mask over [T,T]. Result broadcasts to [B,H,T,T].
    B, T = tgt.shape
    pad = (tgt != pad_id).unsqueeze(1).unsqueeze(1)  # [B,1,1,T]
    causal = make_subsequent_mask(T, device=tgt.device)  # [1,1,T,T]
    return pad & causal


# -------------------------
# Tiny smoke test (run file to verify shapes)
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
