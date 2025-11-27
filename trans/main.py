from __future__ import annotations
from trans.generate import generate_from_checkpoint
import argparse, os, torch, json
from trans.bpe_tokenizer import BPETokenizer
from trans.model import ModelConfig, GPT
from trans.utils import set_seed, sample
from trans.train import TrainConfig, run_training
from trans.trainning.benchmark import benchmark_training
from trans.trainning.profile import profile_training, profile_training_precision

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

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
    ap.add_argument("--estimate-training", "--estimate_training", dest="estimate_training", action="store_true", help="Print training memory + time estimate and exit")
    ap.add_argument("--benchmark", action="store_true", help="Run training benchmark (warmup+timed steps) and exit")
    ap.add_argument("--benchmark_compare", action="store_true",
                    help="Benchmark FP32 vs Mixed Precision (bf16) and print results")
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
    args = ap.parse_args()

    set_seed(args.seed)

    # For CUDA, allow faster matmul kernels (TF32, etc.). No effect on MPS.
    if args.device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
            print("[config] set float32 matmul precision to 'high' for CUDA")
        except Exception as e:
            print(f"[config] could not set matmul precision: {e}")

    if args.estimate_training:
        rep = run_probe(args)
        print("\n=== Training Memory & Time Estimate ===")
        print(f"Device: {rep['device']} ({rep['cuda_name']})")
        print(f"Model params: {rep['params']:,}")
        ps = rep['param_state_bytes_est']
        peak = rep['peak_alloc_bytes_measured']
        total = rep['gpu_total_bytes']
        print(f"Param+grad+optimizer (est): {sizeof_fmt(ps)} (~16 B/param)")
        if peak:
            print(f"Peak CUDA alloc (measured): {sizeof_fmt(peak)}")
            print(f"GPU total memory: {sizeof_fmt(total)}")
            print(f"Fits during measured step? {'YES' if peak < total else 'NO'}")
        else:
            print("CUDA not available: peak alloc not measured.")
        print(f"\nBatch={rep['batch']}, Seq={rep['seq_len']}, d={rep['d_model']}, layers={rep['layers']}, heads={rep['heads']}, vocab={rep['vocab_size']}")
        print(f"Avg step time: {rep['avg_step_time_sec']*1000:.2f} ms/step  (measured)")
        print(f"Est total time for {rep['steps']} steps: {pretty_time(rep['est_total_time_sec'])}")
        print("Note: Estimates depend on kernels/AMP; use as a ballpark.")
        return

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
             use_bfloat16=(args.device in ("cuda", "mps")),
         )

         print(f"[profile] running torch.profiler for {args.profile_steps} steps on device={args.device}")
         profile_training(
             model,
             tok,
             text,
             args.device,
             tcfg,
             num_workers=0,
             profile_steps=args.profile_steps,
         )
         return

    if args.benchmark_compare:
        if not args.train_txt:
            raise SystemExit("--train_txt is required for --benchmark_compare")

        text = load_text(args.train_txt)
        if not os.path.exists(args.tokenizer):
            print("[BPE] training tokenizer for benchmark compare...")
            tok = BPETokenizer.train([text], vocab_size=args.vocab_size, progress=True)
            tok.to_json(args.tokenizer)
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

        # ---- FP32 model ----
        model_fp32 = GPT(cfg).to(args.device)
        tcfg_fp32 = TrainConfig(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            warmup=args.warmup,
            weight_decay=args.wd,
            grad_clip=args.clip,
            sample_every=0,
            sample_prompt=args.sample_prompt,
            use_bfloat16=False,
        )
        print("[bench-compare] FP32...")
        res_fp32 = benchmark_training(
            model_fp32, tok, text, args.device,
            tcfg_fp32, warmup_steps=args.warmup_steps,
            bench_steps=args.measure_steps, num_workers=0, return_stats=True
        )

        # ---- Mixed Precision (bf16) model ----
        model_mp = GPT(cfg).to(args.device)
        tcfg_mp = TrainConfig(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            warmup=args.warmup,
            weight_decay=args.wd,
            grad_clip=args.clip,
            sample_every=0,
            sample_prompt=args.sample_prompt,
            use_bfloat16=True,
        )
        print("[bench-compare] Mixed Precision (bf16)...")
        res_mp = benchmark_training(
            model_mp, tok, text, args.device,
            tcfg_mp, warmup_steps=args.warmup_steps,
            bench_steps=args.measure_steps, num_workers=0, return_stats=True
        )

        print("\n=== FP32 vs Mixed Precision ===")
        print(f"FP32:  mean_ms={res_fp32['mean_ms']:.2f}  tok/s={res_fp32['toks_per_s']:.1f}")
        print(f"MP:    mean_ms={res_mp['mean_ms']:.2f}  tok/s={res_mp['toks_per_s']:.1f}")
        return
    
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
            use_bfloat16=(args.device in ("cuda", "mps")),
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
            num_workers=0,
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
            use_bfloat16=(args.device in ("cuda", "mps")),
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
            num_workers=0,
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

    # Prepare model
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=args.d_model, n_layer=args.layers, n_head=args.heads, ff_mult=args.ff_mult, seq_len=args.seq_len, dropout=args.dropout)
    model = GPT(cfg).to(args.device)

    if args.compile_model and hasattr(torch, "compile"):
        try:
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
        use_bfloat16=(args.device in ("cuda", "mps")),
    )
    run_training(model, tok, text, args.device, tcfg, num_workers=0, resume=resume_obj, save_path=args.save)

    # Final sample
    print("\n[final sample]\n" + sample(model, tok, args.sample_prompt, 200))
    



if __name__ == "__main__":
    main()
