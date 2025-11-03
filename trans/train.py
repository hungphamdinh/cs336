from __future__ import annotations
from dataclasses import dataclass
import math, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step+1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    min_lr = base_lr * 0.1
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))

def run_training(model: GPT, tok: BPETokenizer, train_text: str, device: str, cfg: TrainConfig, num_workers: int = 0, resume: dict|None=None, save_path: str = "ckpt.pt"):
    loader = build_loader(train_text, tok, cfg.seq_len, cfg.batch_size, shuffle=True, num_workers=num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_step = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        start_step = resume.get("step", 0)
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))
    t0 = time.time()
    step = start_step
    while step < cfg.steps:
        for xb,yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            lr = cosine_lr(step, cfg.steps, cfg.warmup, cfg.lr)
            for pg in opt.param_groups: pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            step += 1
            if step % 50 == 0:
                dt = time.time() - t0; t0 = time.time()
                print(f"step={step}/{cfg.steps} loss={float(loss):.4f} lr={lr:.2e} t/50={dt:.2f}s")
            if cfg.sample_every and step % cfg.sample_every == 0:
                print("\n[sample]\n" + sample(model, tok, cfg.sample_prompt, 200)[:300] + "\n")
            if step % 500 == 0 or step == cfg.steps:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": model.cfg.__dict__}, save_path)
                print(f"[ckpt] saved -> {save_path}")
            if step >= cfg.steps: break
