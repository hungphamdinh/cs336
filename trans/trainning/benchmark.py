from __future__ import annotations
import math
import time
import torch

from trans.model import GPT
from trans.bpe_tokenizer import BPETokenizer
from trans.train import (
    TrainConfig,
    build_batch_iterator,
    create_optimizer,
    create_grad_scaler,
    train_step,
)


def _run_benchmark_once(
    model: GPT,
    tok: BPETokenizer,
    train_text: str,
    device: str,
    cfg: TrainConfig,
    warmup_steps: int,
    bench_steps: int,
    num_workers: int,
) -> tuple[float, float, float]:
    """
    Internal helper to run a single benchmark with the given TrainConfig.
    Returns (mean_time_sec, std_time_sec, tokens_per_second).
    """
    batch_iter = build_batch_iterator(train_text, tok, cfg, device, num_workers)
    opt = create_optimizer(model, cfg)
    model.train()
    scaler = create_grad_scaler(device)

    # Warmup phase: run warmup_steps training steps without timing
    warm = 0
    while warm < warmup_steps:
        xb, yb = next(batch_iter)
        _ = train_step(model, opt, scaler, xb, yb, device, cfg)
        warm += 1

    # Timed phase: run bench_steps training steps and measure per-step wall-clock time
    step_times: list[float] = []
    bench = 0
    while bench < bench_steps:
        xb, yb = next(batch_iter)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        _ = train_step(model, opt, scaler, xb, yb, device, cfg)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()
        step_times.append(t1 - t0)
        bench += 1

    total_time = sum(step_times)
    mean_time = total_time / bench_steps
    tokens_per_second = (cfg.batch_size * cfg.seq_len) / mean_time
    std_time = math.sqrt(
        sum((t - mean_time) ** 2 for t in step_times) / max(1, bench_steps - 1)
    )
    return mean_time, std_time, tokens_per_second


def benchmark_training(
    model: GPT,
    tok: BPETokenizer,
    train_text: str,
    device: str,
    cfg: TrainConfig,
    warmup_steps: int = 20,
    bench_steps: int = 100,
    num_workers: int = 0,
    return_stats: bool = False,
):
    """
    Run a benchmark training loop (warmup + measured steps) and optionally return stats.
    """
    mean_time, std_time, tokens_per_second = _run_benchmark_once(
        model, tok, train_text, device, cfg, warmup_steps, bench_steps, num_workers
    )
    ms_per_step = mean_time * 1000.0
    std_ms = std_time * 1000.0
    print(
        f"[bench] steps={bench_steps} mean_ms={ms_per_step:.2f} std_ms={std_ms:.2f} tok/s={tokens_per_second:.1f}"
    )
    if return_stats:
        return {
            "mean_time": mean_time,
            "std_time": std_time,
            "mean_ms": ms_per_step,
            "std_ms": std_ms,
            "toks_per_s": tokens_per_second,
        }


def benchmark_training_precision(
    model: GPT,
    tok: BPETokenizer,
    train_text: str,
    device: str,
    cfg: TrainConfig,
    warmup_steps: int = 20,
    bench_steps: int = 100,
    num_workers: int = 0,
    use_bfloat16: bool = True,
):
    """
    Benchmark training under a specific precision mode.

    - If use_bfloat16 is True and device supports it (cuda/mps),
      training will run with mixed-precision (bf16/fp16).
    - If use_bfloat16 is False, training will run in pure fp32
      (autocast disabled on mps/cuda via cfg.use_bfloat16=False).

    This is useful to compare FP32 vs mixed-precision performance.
    """
    # Clone the TrainConfig so we don't mutate the caller's object
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
    print(f"[bench:{mode}] warmup_steps={warmup_steps}, measure_steps={bench_steps}")
    mean_time, std_time, tokens_per_second = _run_benchmark_once(
        model, tok, train_text, device, cfg_local, warmup_steps, bench_steps, num_workers
    )
    ms_per_step = mean_time * 1000.0
    std_ms = std_time * 1000.0
    print(
        f"[bench:{mode}] steps={bench_steps} mean_ms={ms_per_step:.2f} std_ms={std_ms:.2f} tok/s={tokens_per_second:.1f}"
    )
