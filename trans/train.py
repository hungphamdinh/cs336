from __future__ import annotations
from dataclasses import dataclass
import math, os, time
import torch
import torch.nn.functional as F
from .model import ModelConfig, GPT
from .bpe_tokenizer import BPETokenizer
from .data import build_loader
from .utils import sample


@dataclass
class TrainConfig:
    seq_len: int = 256
    batch_size: int = 32
    steps: int = 2000
    lr: float = 3e-4
    warmup: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    sample_every: int = 0
    sample_prompt: str = "To be, or not to be"
    use_bfloat16: bool = False
    # If True and device != 'cpu', build batches directly on device from the full tokenized text
    in_memory_batches: bool = True

def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step+1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    min_lr = base_lr * 0.1
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))

def create_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

def create_grad_scaler(device: str) -> torch.amp.GradScaler:
    # Use new torch.amp API; enable scaling only on CUDA
    return torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

def compute_loss(model: GPT, xb: torch.Tensor, yb: torch.Tensor, device: str, cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute logits and loss.

    Assumes xb and yb are already on the correct device.
    """
    if device.startswith("cuda"):
        # Decide which AMP dtype to use on CUDA.
        # - If cfg.use_bfloat16 is False: run in pure fp32 (no autocast).
        # - If cfg.use_bfloat16 is True and the GPU supports bf16: use bf16 autocast.
        # - If cfg.use_bfloat16 is True but bf16 is not supported (e.g., T4): fall back to fp16 autocast.
        supports_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

        if cfg.use_bfloat16 and supports_bf16:
            amp_dtype = torch.bfloat16
            use_autocast = True
        elif cfg.use_bfloat16:
            amp_dtype = torch.float16
            use_autocast = True
        else:
            amp_dtype = None
            use_autocast = False

        if use_autocast:
            with torch.amp.autocast(device_type="cuda", enabled=True, dtype=amp_dtype):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
    elif device == "mps":
        # On Apple M1/M2 GPUs, use bf16 autocast if cfg.use_bfloat16, else run in fp32
        if cfg.use_bfloat16:
            with torch.amp.autocast(device_type="mps", enabled=True, dtype=torch.bfloat16):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
    else:
        # On pure CPU or other devices, run in regular fp32 for stability
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
    return loss, logits

def train_step(
    model: GPT,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: str,
    cfg: TrainConfig,
) -> torch.Tensor:
    opt.zero_grad(set_to_none=True)
    loss, _ = compute_loss(model, xb, yb, device, cfg)
    scaler.scale(loss).backward()
    if cfg.grad_clip:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(opt)
    scaler.update()
    return loss

def update_lr(opt: torch.optim.Optimizer, step: int, cfg: TrainConfig) -> float:
    lr = cosine_lr(step, cfg.steps, cfg.warmup, cfg.lr)
    for pg in opt.param_groups:
        pg["lr"] = lr
    return lr

def build_in_memory_batch_iterator(
    train_text: str,
    tok: BPETokenizer,
    seq_len: int,
    batch_size: int,
    device: str,
):
    """
    Build an infinite iterator that yields (xb, yb) batches directly on the target device.

    The entire corpus is tokenized once and stored on `device`, and each batch samples
    random starting positions into this long token sequence.
    """
    # Encode full corpus once
    ids = torch.tensor(tok.encode(train_text), dtype=torch.long, device=device)
    # We need at least seq_len+1 tokens to form input/target sequences
    max_start = ids.size(0) - (seq_len + 1)
    if max_start <= 0:
        raise ValueError(f"Not enough tokens ({ids.size(0)}) for seq_len={seq_len}")

    # Precompute offsets [0, 1, ..., seq_len-1] on device
    offsets = torch.arange(seq_len, device=device)

    def iterator():
        while True:
            # Sample random start indices on device
            starts = torch.randint(0, max_start, (batch_size,), device=device)
            idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
            xb = ids[idx]               # (batch_size, seq_len)
            yb = ids[idx + 1]           # next-token targets
            return_batch = (xb, yb)
            yield return_batch

    return iterator()

def build_batch_iterator(
    train_text: str,
    tok: BPETokenizer,
    cfg: TrainConfig,
    device: str,
    num_workers: int,
):
    """
    Build an infinite iterator that yields (xb, yb) already on the target device.

    - If cfg.in_memory_batches and device != 'cpu', use a single on-device tensor and
      sample from it (no per-batch .to(device) copies).
    - Otherwise, fall back to DataLoader and move each batch to device once.
    """
    if cfg.in_memory_batches and device != "cpu":
        return build_in_memory_batch_iterator(
            train_text,
            tok,
            cfg.seq_len,
            cfg.batch_size,
            device,
        )

    # Fallback: use DataLoader and move each batch to device
    loader = build_loader(
        train_text,
        tok,
        cfg.seq_len,
        cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    def iterator():
        while True:
            for xb, yb in loader:
                yield xb.to(device), yb.to(device)

    return iterator()

def run_training(model: GPT, tok: BPETokenizer, train_text: str, device: str, cfg: TrainConfig, num_workers: int = 0, resume: dict|None=None, save_path: str = "ckpt.pt"):
    batch_iter = build_batch_iterator(train_text, tok, cfg, device, num_workers)
    opt = create_optimizer(model, cfg)
    start_step = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        start_step = resume.get("step", 0)
    model.train()
    scaler = create_grad_scaler(device)
    t0 = time.time()
    step = start_step
    while step < cfg.steps:
        xb, yb = next(batch_iter)
        lr = update_lr(opt, step, cfg)
        loss = train_step(model, opt, scaler, xb, yb, device, cfg)
        step += 1
        if step % 50 == 0:
            dt = time.time() - t0; t0 = time.time()
            print(f"step={step}/{cfg.steps} loss={float(loss):.4f} lr={lr:.2e} t/50={dt:.2f}s")
        if cfg.sample_every and step % cfg.sample_every == 0:
            print("\n[sample]\n" + sample(model, tok, cfg.sample_prompt, 200)[:300] + "\n")
        if save_path is not None and (step % 500 == 0 or step == cfg.steps):
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "cfg": model.cfg.__dict__,
                },
                save_path,
            )
            print(f"[ckpt] saved -> {save_path}")
