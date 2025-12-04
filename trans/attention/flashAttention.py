import math
import torch

try:
    # Optional FlashAttention-2 integration (only works on supported NVIDIA GPUs, e.g. A100/H100).
    # On unsupported setups (Mac M1/M2, older GPUs, or if the package isn't installed),
    # this import will fail and we will silently fall back to the local attention path.
    from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn_func = None  # type: ignore[assignment]
    _HAS_FLASH_ATTN = False


def can_use_flash_attention(q: torch.Tensor) -> bool:
    """
    Return True if we can safely run FlashAttention-2 on this tensor.

    Conditions:
    - flash-attn is importable
    - running on CUDA
    - GPU compute capability is sm80+ (A100/H100 class)
    - dtype is fp16 or bf16 (otherwise we fall back)
    """
    if not _HAS_FLASH_ATTN:
        return False
    if not q.is_cuda:
        return False
    try:
        major, minor = torch.cuda.get_device_capability(q.device)
    except Exception:
        return False
    # FlashAttention-2 supports sm80+; skip on T4 / older (sm75 etc.).
    if 10 * major + minor < 80:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


def flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dropout_p: float,
    training: bool,
) -> torch.Tensor | None:
    """
    Run FlashAttention-2 if available; otherwise return None so the caller can fall back.

    Q, K, V: [B, H, L, D]
    dropout_p: attention dropout probability from the calling module
    training: whether the caller is in training mode
    """
    if not can_use_flash_attention(Q):
        return None

    B, H, L, D = Q.shape

    # flash_attn_func expects [B*H, L, D] and returns the same shape.
    q = Q.reshape(B * H, L, D).contiguous()
    k = K.reshape(B * H, L, D).contiguous()
    v = V.reshape(B * H, L, D).contiguous()

    # Only apply dropout during training and if p > 0.
    attn_dropout = dropout_p if training and dropout_p > 0.0 else 0.0

    try:
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=attn_dropout,
            softmax_scale=None,
            causal=True,
        )  # [B*H, L, D]
    except Exception:
        # If anything goes wrong (unsupported dtype, device, etc.), signal fallback.
        return None

    return out.reshape(B, H, L, D)
