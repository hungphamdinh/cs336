import math
import torch
import torch.nn.functional as F



def local_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal_mask: torch.Tensor,
    dropout_module,
) -> torch.Tensor:
    """
    Compute causal self-attention using PyTorch SDPA when possible,
    otherwise fall back to an explicit matmul + softmax implementation.

    Arguments:
    - Q, K, V: [B, H, L, D]
    - causal_mask: [1, 1, L, L] (or broadcastable to [B,H,L,L])
    - dropout_module: the Dropout instance from the calling attention module
      (we use its p and training flag to decide attention dropout).
    """
    B, H, L, D = Q.shape
    use_sdp = hasattr(F, "scaled_dot_product_attention")

    # Attention dropout probability; only applied during training.
    dropout_p = dropout_module.p if dropout_module.training and dropout_module.p > 0.0 else 0.0

    if use_sdp:
        qkv_dtype = Q.dtype
        try:
            # scaled_dot_product_attention currently supports {Half, Float} on most backends.
            # If we're in a different dtype (e.g., bf16 on some GPUs), promote to float32.
            if qkv_dtype not in (torch.float16, torch.float32):
                Q_sdp = Q.to(torch.float32)
                K_sdp = K.to(torch.float32)
                V_sdp = V.to(torch.float32)
            else:
                Q_sdp, K_sdp, V_sdp = Q, K, V

            attn = F.scaled_dot_product_attention(
                Q_sdp,
                K_sdp,
                V_sdp,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )  # [B,H,L,D]

            # Cast back to the original dtype if we promoted.
            if attn.dtype != qkv_dtype:
                attn = attn.to(qkv_dtype)

            return attn
        except RuntimeError:
            # If SDPA fails (e.g., unsupported dtype/device combo), fall through to manual path.
            pass

    # Manual attention path: QK^T / sqrt(d), apply causal mask, softmax, dropout, then multiply by V.
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)  # [B,H,L,L]
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    P = F.softmax(scores, dim=-1)
    P = dropout_module(P)
    return P @ V  # [B,H,L,D]
