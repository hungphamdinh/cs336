from __future__ import annotations
from trans.generate import generate_from_checkpoint
import argparse, os, torch, json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from trans.bpe_tokenizer import BPETokenizer
from trans.model import ModelConfig, GPT
from trans.utils import set_seed, sample
from trans.train import TrainConfig, run_training
from trans.trainning.benchmark import benchmark_training
from trans.trainning.profile import profile_training, profile_training_precision

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Helper for distributed info
def get_dist_info():
    """
    Inspect environment variables (as set by torchrun) and return (rank, world_size, is_ddp).
    If not running under distributed launch, returns (0, 1, False).
    """
    if not dist.is_available():
        return 0, 1, False
    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    if rank_env is None or world_env is None:
        return 0, 1, False
    rank = int(rank_env)
    world_size = int(world_env)
    return rank, world_size, world_size > 1

def main():
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    # If CUDA is available, prefer Flash / memory-efficient scaled dot-product attention
    # (used by F.scaled_dot_product_attention) when possible.
    if default_device == "cuda" and hasattr(torch.backends, "cuda"):
        try:
            # Try to enable Flash / mem-efficient SDPA, but keep math kernel enabled
            # so that older GPUs (e.g., T4 on Kaggle, sm75) still have a valid fallback.
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                # IMPORTANT: keep math kernel ON, otherwise some (arch, dtype)
                # combinations will have *no* available kernel and raise:
                # "RuntimeError: No available kernel. Aborting execution."
                torch.backends.cuda.enable_math_sdp(True)
            print("[config] configured SDPA backends on CUDA (flash/mem-efficient + math fallback)")
        except Exception as e:
            print(f"[config] could not configure SDPA backends: {e}")

    ap = argparse.ArgumentParser(description="Transformer training with byte-level BPE")
    ap.add_argument("--train_txt", type=str, help="Path to training text (raw UTF-8)")
    ap.add_argument("--tokenizer", type=str, default="tokenizer.json")
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sample_every", type=int, default=0)
    ap.add_argument("--sample_prompt", type=str, default="Once upon a time")
    ap.add_argument("--device", type=str, default=default_device)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--save", type=str, default="ckpt.pt")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--generate", action="store_true", help="If set, run text generation from a checkpoint and exit")
    ap.add_argument("--ckpt", type=str, default="ckpt.pt", help="Path to checkpoint for --generate")
    ap.add_argument("--gen_prompt", type=str, default="Once upon a time", help="Prompt text for --generate")
    ap.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate for --generate")
    ap.add_argument("--benchmark", action="store_true", help="Run training benchmark (warmup+timed steps) and exit")
    ap.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps for --estimate-training probe")
    ap.add_argument("--measure_steps", type=int, default=10, help="Measurement steps for --estimate-training probe")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Profile a few training steps with torch.profiler and exit",
    )
    ap.add_argument(
        "--profile_steps",
        type=int,
        default=20,
        help="How many training steps to run under profiler for --profile",
    )
    ap.add_argument(
        "--profile_precision",
        action="store_true",
        help="Profile training under a specific precision (fp32 or bf16) and exit",
    )
    ap.add_argument(
        "--precision_mode",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
        help="Precision mode for --profile_precision (fp32 or bf16)",
    )
    ap.add_argument(
        "--compile_model",
        action="store_true",
        help="Use torch.compile() to optimize model execution (PyTorch 2.x)",
    )
    ap.add_argument(
        "--precision",
        type=str,
        choices=["auto", "fp32", "bf16"],
        default="auto",
        help="Training precision: auto (default), fp32, or bf16",
    )
    ap.add_argument(
        "--benchmark_compile_compare",
        action="store_true",
        help="Benchmark with and without torch.compile() and compare",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers for CPU batch building when using DataLoader-based batches.",
    )
    ap.add_argument(
        "--in_memory_batches",
        dest="in_memory_batches",
        action="store_true",
        help="Use on-device in-memory random sampling batches where supported (default on GPU/MPS).",
    )
    ap.add_argument(
        "--no_in_memory_batches",
        dest="in_memory_batches",
        action="store_false",
        help="Disable in-memory batches; always use DataLoader + CPU workers (applies to both CPU and GPU/MPS).",
    )
    ap.set_defaults(in_memory_batches=True)
    args = ap.parse_args()

    set_seed(args.seed)

    rank, world_size, is_ddp = get_dist_info()
    # Allow DDP **only** for CPU/MPS. Disable GPU DDP completely.
    if is_ddp and (args.device == "cpu" or args.device == "mps"):
        backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        set_seed(args.seed + rank)
        if rank == 0:
            print(f"[ddp] (CPU/MPS only) initialized process group backend=gloo rank={rank} world_size={world_size}")
    else:
        # Disable DDP entirely for CUDA
        is_ddp = False
        rank, world_size = 0, 1

    # Decide training precision in a device-agnostic way:
    # - auto: use bf16 on CUDA (AMP), fp32 elsewhere
    # - fp32: always fp32
    # - bf16: try to use bf16 where supported
    if args.precision == "fp32":
        train_use_bfloat16 = False
    elif args.precision == "bf16":
        train_use_bfloat16 = True
    else:  # "auto"
        train_use_bfloat16 = (args.device == "cuda")

    print(f"[config] device={args.device}, precision={'bf16' if train_use_bfloat16 else 'fp32'}")

    # For CUDA, allow faster matmul kernels (TF32, etc.). No effect on MPS.
    if args.device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
            print("[config] set float32 matmul precision to 'high' for CUDA")
        except Exception as e:
            print(f"[config] could not set matmul precision: {e}")

    # Removed estimate_training block

    if args.profile:
         if not args.train_txt:
             raise SystemExit("--train_txt is required for --profile")

         text = load_text(args.train_txt)

         # Load or train tokenizer (same logic as training)
         if args.resume is None and not os.path.exists(args.tokenizer):
             print("[BPE] training tokenizer for profiling...")
             tok = BPETokenizer.train([text], vocab_size=args.vocab_size, progress=True)
             tok.to_json(args.tokenizer)
             print(f"[BPE] saved -> {args.tokenizer} (vocab={tok.vocab_size})")
         else:
             tok = BPETokenizer.from_json(args.tokenizer)

         # Prepare model config and model
         cfg = ModelConfig(
             vocab_size=tok.vocab_size,
             d_model=args.d_model,
             n_layer=args.layers,
             n_head=args.heads,
             ff_mult=args.ff_mult,
             seq_len=args.seq_len,
             dropout=args.dropout,
         )
         model = GPT(cfg).to(args.device)

         if args.compile_model and hasattr(torch, "compile"):
             try:
                 model = torch.compile(model)
                 print("[compile] model compiled with torch.compile()")
             except Exception as e:
                 print(f"[compile] torch.compile failed: {e}")

         # Training config (same defaults as normal training)
         tcfg = TrainConfig(
             seq_len=args.seq_len,
             batch_size=args.batch_size,
             steps=args.steps,
             lr=args.lr,
             warmup=args.warmup,
             weight_decay=args.wd,
             grad_clip=args.clip,
             sample_every=0,
             sample_prompt=args.sample_prompt,
             use_bfloat16=train_use_bfloat16,
             in_memory_batches=args.in_memory_batches,
         )

         print(f"[profile] running torch.profiler for {args.profile_steps} steps on device={args.device}")
         profile_training(
             model,
             tok,
             text,
             args.device,
             tcfg,
             num_workers=args.num_workers,
             profile_steps=args.profile_steps,
         )
         return

    # Removed benchmark_compare block (FP32 vs Mixed Precision benchmark)
    
    if args.profile_precision:
        if not args.train_txt:
            raise SystemExit("--train_txt is required for --profile_precision")

        text = load_text(args.train_txt)

        # Load or train tokenizer (same logic as training)
        if args.resume is None and not os.path.exists(args.tokenizer):
            print("[BPE] training tokenizer for precision profiling...")
            tok = BPETokenizer.train([text], vocab_size=args.vocab_size, progress=True)
            tok.to_json(args.tokenizer)
            print(f"[BPE] saved -> {args.tokenizer} (vocab={tok.vocab_size})")
        else:
            tok = BPETokenizer.from_json(args.tokenizer)

        cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            d_model=args.d_model,
            n_layer=args.layers,
            n_head=args.heads,
            ff_mult=args.ff_mult,
            seq_len=args.seq_len,
            dropout=args.dropout,
        )
        model = GPT(cfg).to(args.device)

        if args.compile_model and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("[compile] model compiled with torch.compile()")
            except Exception as e:
                print(f"[compile] torch.compile failed: {e}")

        # Base training config; actual precision is controlled by profile_training_precision
        tcfg = TrainConfig(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            warmup=args.warmup,
            weight_decay=args.wd,
            grad_clip=args.clip,
            sample_every=0,
            sample_prompt=args.sample_prompt,
            use_bfloat16=train_use_bfloat16,
            in_memory_batches=args.in_memory_batches,
        )

        use_bfloat16 = args.precision_mode == "bf16"
        print(
            f"[profile_precision] running torch.profiler for {args.profile_steps} steps on device={args.device} mode={args.precision_mode}"
        )
        profile_training_precision(
            model,
            tok,
            text,
            args.device,
            tcfg,
            profile_steps=args.profile_steps,
            num_workers=args.num_workers,
            use_bfloat16=use_bfloat16,
        )
        return


    if args.benchmark:
        if not args.train_txt:
            raise SystemExit("--train_txt is required for --benchmark")
        # Load text (same as normal training)
        text = load_text(args.train_txt)
        # Load or train tokenizer (same logic as below, but avoid duplicate code by reusing args.tokenizer)
        if not os.path.exists(args.tokenizer):
            print("[BPE] training tokenizer for benchmark...")
            tok = BPETokenizer.train([text], vocab_size=args.vocab_size, progress=True)
            tok.to_json(args.tokenizer)
            print(f"[BPE] saved -> {args.tokenizer} (vocab={tok.vocab_size})")
        else:
            tok = BPETokenizer.from_json(args.tokenizer)
        # Prepare model config and model
        cfg = ModelConfig(
            vocab_size=tok.vocab_size,
            d_model=args.d_model,
            n_layer=args.layers,
            n_head=args.heads,
            ff_mult=args.ff_mult,
            seq_len=args.seq_len,
            dropout=args.dropout,
        )
        model = GPT(cfg).to(args.device)

        if args.compile_model and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("[compile] model compiled with torch.compile()")
            except Exception as e:
                print(f"[compile] torch.compile failed: {e}")

        tcfg = TrainConfig(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            warmup=args.warmup,
            weight_decay=args.wd,
            grad_clip=args.clip,
            sample_every=0,
            sample_prompt=args.sample_prompt,
            use_bfloat16=train_use_bfloat16,
            in_memory_batches=args.in_memory_batches,
        )
        print(f"[bench] running benchmark: warmup_steps={args.warmup_steps}, measure_steps={args.measure_steps}")
        benchmark_training(
            model,
            tok,
            text,
            args.device,
            tcfg,
            warmup_steps=args.warmup_steps,
            bench_steps=args.measure_steps,
            num_workers=args.num_workers,
            return_stats=False,
        )
        return

    # If generation mode is requested, run it and exit
    if args.generate:
        generate_from_checkpoint(
            ckpt_path=args.ckpt,
            tokenizer_path=args.tokenizer,
            prompt=args.gen_prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        return

    # Load or train tokenizer
    if args.resume is None and not os.path.exists(args.tokenizer):
        if not args.train_txt: raise SystemExit("--train_txt is required to train tokenizer")
        print("[BPE] training tokenizer...")
        text_for_bpe = load_text(args.train_txt)
        tok = BPETokenizer.train([text_for_bpe], vocab_size=args.vocab_size, progress=True)
        tok.to_json(args.tokenizer)
        print(f"[BPE] saved -> {args.tokenizer} (vocab={tok.vocab_size})")
    else:
        tok = BPETokenizer.from_json(args.tokenizer)

    # Prepare model (single-process or wrapped in DDP depending on environment)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layer=args.layers,
        n_head=args.heads,
        ff_mult=args.ff_mult,
        seq_len=args.seq_len,
        dropout=args.dropout,
    )
    model = GPT(cfg)

    # Decide which device this rank should actually train on.
    train_device = args.device
    if is_ddp:
        # Only CPU/MPS DDP allowed
        train_device = args.device
        model = model.to(train_device)
        model = DDP(model)  # no device_ids needed for CPU/MPS
        if rank == 0:
            print(f"[ddp] using CPU/MPS DDP on device={train_device}")
    else:
        # GPU always runs single-process
        model = model.to(train_device)

    if args.compile_model and hasattr(torch, "compile"):
        try:
            # torch.compile should wrap the *local* model (DDP-wrapped or plain).
            model = torch.compile(model)
            print("[compile] model compiled with torch.compile()")
        except Exception as e:
            print(f"[compile] torch.compile failed: {e}")

    # Resume if requested
    resume_obj = None
    if args.resume and os.path.exists(args.resume):
        print(f"[ckpt] loading {args.resume}")
        resume_obj = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_obj["model"])

    # Train text
    if not args.train_txt:
        # tiny fallback corpus
        text = ("From fairest creatures we desire increase,\n" * 200)
    else:
        text = load_text(args.train_txt)
        
    

    # Train
    tcfg = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        warmup=args.warmup,
        weight_decay=args.wd,
        grad_clip=args.clip,
        sample_every=args.sample_every,
        sample_prompt=args.sample_prompt,
        use_bfloat16=train_use_bfloat16,
        in_memory_batches=args.in_memory_batches,
    )

    # Only rank 0 should write checkpoints to disk; other ranks pass save_path=None
    save_path = args.save if (rank == 0) else None

    run_training(
        model,
        tok,
        text,
        train_device,
        tcfg,
        num_workers=args.num_workers,
        resume=resume_obj,
        save_path=save_path,
    )

    # Final sample: only print from rank 0 to avoid duplicate output.
    if rank == 0:
        print("\n[final sample]\n" + sample(model, tok, args.sample_prompt, 200))
    



if __name__ == "__main__":
    main()
