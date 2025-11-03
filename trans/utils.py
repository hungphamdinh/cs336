from __future__ import annotations
import random
import torch
import torch.nn.functional as F
from .bpe_tokenizer import BPETokenizer
from .model import GPT

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def sample(model: GPT, tok: BPETokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0) -> str:
    model.eval()
    device = next(model.parameters()).device
    ids = tok.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.cfg.seq_len:]
        logits = model(x_cond)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id], dim=1)
    return tok.decode(x[0].tolist())
