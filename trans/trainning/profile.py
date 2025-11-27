from __future__ import annotations
import os
import torch
from torch.profiler import profile, ProfilerActivity

from trans.model import GPT
from trans.bpe_tokenizer import BPETokenizer
from trans.train import (
    TrainConfig,
    build_batch_iterator,
    create_optimizer,
    create_grad_scaler,
    train_step,
)


def profile_training(
    model: GPT,
    tok: BPETokenizer,
    train_text: str,
    device: str,
    cfg: TrainConfig,
    num_workers: int = 0,
    profile_steps: int = 20,
    trace_dir: str = "profiles",
):
    """
    Run a few training steps under torch.profiler to find bottlenecks.

    - Uses the same train_step() as normal training/benchmark.
    - On M1 (device='mps'), profiles CPU activity; on CUDA, CPU+CUDA.
    - Writes:
      * a text summary to stdout
      * a Chrome trace to profiles/trace.json
    """
    batch_iter = build_batch_iterator(
        train_text,
        tok,
        cfg,
        device,
        num_workers,
    )
    opt = create_optimizer(model, cfg)
    scaler = create_grad_scaler(device)
    model.train()

    # Choose what to profile: always CPU, plus CUDA if available.
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "trace.json")

    step = 0
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        while step < profile_steps:
            xb, yb = next(batch_iter)
            # Single training step (forward + backward + optimizer)
            _ = train_step(model, opt, scaler, xb, yb, device, cfg)

            prof.step()
            step += 1

    # Print a human-readable summary to the console
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=30,
        )
    )

    # Export Chrome trace for visualization in chrome://tracing or TensorBoard
    prof.export_chrome_trace(trace_path)
    print(f"[profile] Chrome trace written to: {trace_path}")
    print("          Open this in chrome://tracing or TensorBoard (Profile plugin).")


# Profile training with a specific precision mode (bf16/fp16 or fp32)
def profile_training_precision(
    model: GPT,
    tok: BPETokenizer,
    train_text: str,
    device: str,
    cfg: TrainConfig,
    profile_steps: int = 20,
    num_workers: int = 0,
    use_bfloat16: bool = True,
    trace_dir: str = "profiles",
):
    cfg_local = TrainConfig(
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        steps=cfg.steps,
        lr=cfg.lr,
        warmup=cfg.warmup,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
        sample_every=cfg.sample_every,
        sample_prompt=cfg.sample_prompt,
        use_bfloat16=use_bfloat16,
        in_memory_batches=cfg.in_memory_batches,
    )

    mode = "mixed (bf16/fp16)" if use_bfloat16 else "fp32"
    print(f"[profile:{mode}] steps={profile_steps}")

    batch_iter = build_batch_iterator(
        train_text, tok, cfg_local, device, num_workers
    )
    opt = create_optimizer(model, cfg_local)
    scaler = create_grad_scaler(device)
    model.train()

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"trace_{mode}.json")

    step = 0
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        while step < profile_steps:
            xb, yb = next(batch_iter)
            _ = train_step(model, opt, scaler, xb, yb, device, cfg_local)
            prof.step()
            step += 1

    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=30,
        )
    )
    prof.export_chrome_trace(trace_path)
    print(f"[profile:{mode}] Chrome trace written to: {trace_path}")
