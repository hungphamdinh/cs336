#!/usr/bin/env python3
# mock_training.py — minimal toy training for your Transformer
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

from Train import Transformer, make_src_mask, make_tgt_mask  # import your code

# ----------------------
# Vocab / tokens
# ----------------------
PAD, BOS, EOS = 0, 1, 2

def build_vocab(symbols: str):
    # symbols are single-char tokens, e.g. "abcdef "
    stoi = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS}
    for i, ch in enumerate(symbols, start=3):
        stoi[ch] = i
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

def encode(txt: str, stoi: dict) -> List[int]:
    return [stoi[c] for c in txt]

def decode(ids: List[int], itos: dict) -> str:
    # strip special tokens
    return "".join(itos[i] for i in ids if i >= 3)

# ----------------------
# Logging helpers
# ----------------------
def log_vocab(stoi, itos):
    print("\n[RAW VOCAB]")
    print({k: v for k, v in stoi.items()})

def log_pairs(name, pairs, k=8):
    print(f"\n[RAW DATA] {name} (showing {min(k, len(pairs))}/{len(pairs)})")
    for i, p in enumerate(pairs[:k]):
        print(f"  {i:02d}: SRC={repr(p.src)} | TGT={repr(p.tgt)}")

def log_batch(src, tgt_in, tgt_out, itos, title="[DATA INPUT] BATCH", k=5):
    print(f"\n{title}")
    print("  shapes:", "src", tuple(src.size()), "tgt_in", tuple(tgt_in.size()), "tgt_out", tuple(tgt_out.size()))
    B = min(k, src.size(0))
    for i in range(B):
        src_txt = decode(src[i].tolist(), itos)
        tgt_in_txt = decode(tgt_in[i].tolist(), itos)
        tgt_out_txt = decode(tgt_out[i].tolist(), itos)
        print(f"  #{i}:")
        print("    src_ids:", src[i].tolist())
        print("    src_txt:", repr(src_txt))
        print("    tgt_in_ids:", tgt_in[i].tolist())
        print("    tgt_in_txt:", repr(tgt_in_txt))
        print("    tgt_out_ids:", tgt_out[i].tolist())
        print("    tgt_out_txt:", repr(tgt_out_txt))

# ----------------------
# Toy parallel data
# ----------------------
@dataclass
class ToyPair:
    src: str
    tgt: str

def make_toy_pairs(n: int, *, min_len=3, max_len=10, symbols="abcd ", task="copy") -> List[ToyPair]:
    pairs = []
    for _ in range(n):
        L = random.randint(min_len, max_len)
        s = "".join(random.choice(symbols) for _ in range(L)).strip()
        s = s if s else "a"  # avoid empty after strip
        if task == "reverse":
            t = s[::-1]
        else:
            t = s
        pairs.append(ToyPair(s, t))
    return pairs

# ----------------------
# Dataset / collate
# ----------------------
class ToySeq2Seq(Dataset):
    def __init__(self, pairs: List[ToyPair], stoi: dict):
        self.pairs = pairs
        self.stoi = stoi

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        src_ids = encode(p.src, self.stoi)                # no BOS/EOS on src
        tgt_ids = encode(p.tgt, self.stoi) + [EOS]        # EOS on target sequence
        tgt_in  = [BOS] + encode(p.tgt, self.stoi)        # BOS-prefixed input to decoder
        return torch.tensor(src_ids), torch.tensor(tgt_in), torch.tensor(tgt_ids)

def pad_to_length(x: List[int], L: int, pad_val=PAD) -> List[int]:
    return x + [pad_val] * (L - len(x))

def collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    srcs, tgt_ins, tgt_outs = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgt_ins)  # tgt_in and tgt_out have same length

    src = torch.tensor([pad_to_length(s.tolist(), max_src) for s in srcs], dtype=torch.long)
    tgt_in = torch.tensor([pad_to_length(t.tolist(), max_tgt) for t in tgt_ins], dtype=torch.long)
    tgt_out = torch.tensor([pad_to_length(t.tolist(), max_tgt) for t in tgt_outs], dtype=torch.long)

    # Masks from your Train.py
    src_mask = make_src_mask(src, pad_id=PAD)       # [B,1,1,S]
    tgt_mask = make_tgt_mask(tgt_in, pad_id=PAD)    # [B,1,T,T]
    return src, tgt_in, tgt_out, src_mask, tgt_mask

# ----------------------
# Greedy decode (for quick sanity checks)
# ----------------------
@torch.no_grad()
def greedy_decode(model: Transformer, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, bos=BOS, eos=EOS):
    device = src.device
    B = src.size(0)
    ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
    for _ in range(max_len):
        tgt_mask = make_tgt_mask(ys, pad_id=PAD)
        logits = model(src, ys, src_mask, tgt_mask)  # [B,T,V]
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # [B,1]
        ys = torch.cat([ys, next_token], dim=1)
        if (next_token == eos).all():
            break
    return ys

# ----------------------
# Train
# ----------------------
def main():
    random.seed(0)
    torch.manual_seed(0)
    debug = True  # set to False to silence logs

    # 1) Build toy data
    symbols = "abcd "  # small alphabet + space
    stoi, itos = build_vocab(symbols)
    if debug:
        log_vocab(stoi, itos)

    train_pairs = make_toy_pairs(1200, task="copy", symbols=symbols)
    valid_pairs = make_toy_pairs(100, task="copy", symbols=symbols)
    if debug:
        log_pairs("TRAIN", train_pairs, k=8)
        log_pairs("VALID", valid_pairs, k=4)

    train_ds = ToySeq2Seq(train_pairs, stoi)
    valid_ds = ToySeq2Seq(valid_pairs, stoi)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate)
    if debug:
        _batch = next(iter(train_loader))
        _src, _tgt_in, _tgt_out, _src_mask, _tgt_mask = _batch
        log_batch(_src, _tgt_in, _tgt_out, itos, title="[DATA INPUT] FIRST TRAIN BATCH", k=5)
        print("  src_mask shape:", tuple(_src_mask.size()), "tgt_mask shape:", tuple(_tgt_mask.size()))

    # 2) Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    V = len(stoi)
    model = Transformer(src_vocab=V, tgt_vocab=V, d_model=128, N=2, h=4, d_ff=256, dropout=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # 3) Train loop
    def run_epoch(loader, train=True):
        model.train(train)
        total, n = 0.0, 0
        for src, tgt_in, tgt_out, src_mask, tgt_mask in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)

            logits = model(src, tgt_in, src_mask, tgt_mask)        # [B,T,V]
            loss = F.cross_entropy(
                logits.reshape(-1, V), tgt_out.reshape(-1), ignore_index=PAD
            )
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item()
            n += 1
        return total / max(1, n)

    for epoch in range(1, 11):
        tr_loss = run_epoch(train_loader, train=True)
        va_loss = run_epoch(valid_loader, train=False)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.3f} | valid loss {va_loss:.3f}")

    # 4) Quick sanity check: decode a few val examples
    model.eval()
    src, tgt_in, tgt_out, src_mask, tgt_mask = next(iter(valid_loader))
    src, src_mask = src.to(device), src_mask.to(device)
    pred = greedy_decode(model, src[:5], src_mask[:5], max_len=tgt_in.size(1)+2)
    final_dump = []
    for i in range(min(5, src.size(0))):
        src_txt = decode(src[i].tolist(), itos)
        tgt_txt = decode(tgt_out[i].tolist(), itos)
        pr_txt  = decode(pred[i].tolist(), itos)
        print(f"SRC: {repr(src_txt)}")
        print(f"TGT: {repr(tgt_txt)}")
        print(f"PRD: {repr(pr_txt)}")
        final_dump.append({"src": src_txt, "tgt": tgt_txt, "pred": pr_txt})
        print("—")

    print("\n[FINAL DATA OUTPUT] sample JSON:")
    print(json.dumps(final_dump, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()