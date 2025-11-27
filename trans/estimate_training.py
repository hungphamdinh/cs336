from __future__ import annotations
import time, sys
import torch
import torch.nn.functional as F
from trans.model import ModelConfig, GPT

def sizeof_fmt(num_bytes: int) -> str:
    # Human‑readable bytes
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} EB"

@torch.no_grad()
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _choose_device(arg_device: str) -> torch.device:
    if arg_device and arg_device != "auto":
        return torch.device(arg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _build_model(args) -> GPT:
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layer=args.layers,
        n_head=args.heads,
        ff_mult=args.ff_mult,
        seq_len=args.seq_len,
        dropout=0.0,
    )
    return GPT(cfg)

def _estimate_param_state_bytes(n_params: int, training: bool = True) -> int:
    # Typical Adam/AdamW in fp32 states during mixed‑precision training:
    # params(4B) + grads(4B) + m(4B) + v(4B) ~= 16B / param
    return n_params * (16 if training else 4)

def pretty_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d: parts.append(f"{int(d)}d")
    if h: parts.append(f"{int(h)}h")
    if m: parts.append(f"{int(m)}m")
    parts.append(f"{s:.1f}s")
    return " ".join(parts)

def run_probe(args) -> dict:
    """
    Build the model with args, run a short dummy train micro‑benchmark,
    and report memory + timing. Safe to call before real training.
    Required args fields:
      - vocab_size, seq_len, batch_size
      - d_model, heads, layers, ff_mult
      - steps (for ETA), device (or 'auto')
    """
    device = _choose_device(getattr(args, "device", "auto"))
    model = _build_model(args).to(device)
    n_params = count_params(model)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Dummy batch (random tokens)
    B, L, V = args.batch_size, args.seq_len, args.vocab_size
    xb = torch.randint(0, V, (B, L), device=device, dtype=torch.long)
    yb = torch.randint(0, V, (B, L), device=device, dtype=torch.long)

    use_cuda = (device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast(enabled=use_cuda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # Reset CUDA stats to measure peak alloc during a real step
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model.train()

    def one_step():
        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(xb)                  # [B, L, V]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        if use_cuda:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return float(loss.detach())

    # Warmup a few iters to stabilize kernels
    warmup = max(3, getattr(args, "warmup", 5))
    measure = max(5, getattr(args, "measure_steps", 10))

    for _ in range(warmup):
        _ = one_step()
        if use_cuda: torch.cuda.synchronize()

    # Measure avg step time over a few iters
    times = []
    for _ in range(measure):
        t0 = time.perf_counter()
        _ = one_step()
        if use_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_step_time = sum(times) / len(times)
    total_time_sec = avg_step_time * getattr(args, "steps", 1)

    # Memory stats
    param_state_bytes = _estimate_param_state_bytes(n_params, training=True)
    peak_alloc = torch.cuda.max_memory_allocated(device) if use_cuda else 0
    total_mem = torch.cuda.get_device_properties(device).total_memory if use_cuda else 0

    return {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if use_cuda else "CPU",
        "params": n_params,
        "param_state_bytes_est": param_state_bytes,
        "peak_alloc_bytes_measured": peak_alloc,
        "gpu_total_bytes": total_mem,
        "batch": B,
        "seq_len": L,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "ff_mult": args.ff_mult,
        "vocab_size": V,
        "avg_step_time_sec": avg_step_time,
        "est_total_time_sec": total_time_sec,
        "steps": getattr(args, "steps", 1),
        "warmup": warmup,
        "measure_steps": measure,
    }
