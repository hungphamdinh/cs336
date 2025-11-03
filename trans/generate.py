

import torch
import os
from trans.bpe_tokenizer import BPETokenizer
from trans.model import ModelConfig, GPT
from trans.utils import sample

# Helper: load a checkpoint and generate output for verification
def generate_from_checkpoint(ckpt_path: str, tokenizer_path: str, prompt: str = "Once upon a time", max_new_tokens: int = 200, device: str = None):
    """Load a saved checkpoint and generate output for verification."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve device gracefully
    if device == "cuda" and not torch.cuda.is_available():
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("[device] CUDA not available. Falling back to MPS.")
            device = "mps"
        else:
            print("[device] CUDA not available. Falling back to CPU.")
            device = "cpu"
    if device == "mps":
        has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not has_mps:
            print("[device] MPS not available. Falling back to CPU.")
            device = "cpu"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    print(f"[load] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise ValueError("Checkpoint missing model config ('cfg').")

    tok = BPETokenizer.from_json(tokenizer_path)
    cfg = ModelConfig(**cfg_dict)
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[generate] prompt: {prompt!r}")
    output = sample(model, tok, prompt, max_new_tokens)
    print(f"\n[output]\n{output}\n")
    return output