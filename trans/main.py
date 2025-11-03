from __future__ import annotations
from trans.generate import generate_from_checkpoint
import argparse, os, torch, json
from trans.bpe_tokenizer import BPETokenizer
from trans.model import ModelConfig, GPT
from trans.utils import set_seed, sample
from trans.train import TrainConfig, run_training

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
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
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--save", type=str, default="ckpt.pt")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--generate", action="store_true", help="If set, run text generation from a checkpoint and exit")
    ap.add_argument("--ckpt", type=str, default="ckpt.pt", help="Path to checkpoint for --generate")
    ap.add_argument("--gen_prompt", type=str, default="Once upon a time", help="Prompt text for --generate")
    ap.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate for --generate")
    args = ap.parse_args()

    set_seed(args.seed)

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
    tcfg = TrainConfig(seq_len=args.seq_len, batch_size=args.batch_size, steps=args.steps, lr=args.lr, warmup=args.warmup, weight_decay=args.wd, grad_clip=args.clip, sample_every=args.sample_every, sample_prompt=args.sample_prompt)
    run_training(model, tok, text, args.device, tcfg, num_workers=0, resume=resume_obj, save_path=args.save)

    # Final sample
    print("\n[final sample]\n" + sample(model, tok, args.sample_prompt, 200))



if __name__ == "__main__":
    main()
