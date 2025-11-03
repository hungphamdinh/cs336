from __future__ import annotations
from typing import Tuple
import math
import torch
from torch.utils.data import Dataset, DataLoader
from .bpe_tokenizer import BPETokenizer

class ByteBPETextDataset(Dataset):
    def __init__(self, text: str, tokenizer: BPETokenizer, seq_len: int):
        self.tok = tokenizer
        self.ids = torch.tensor(self.tok.encode(text), dtype=torch.long)
        self.seq_len = seq_len
        if len(self.ids) < seq_len + 1:
            reps = math.ceil((seq_len + 1) / max(1, len(self.ids)))
            self.ids = self.ids.repeat(reps)
        self.ids = self.ids[: ((len(self.ids) - 1)//seq_len)*seq_len + 1]

    def __len__(self):
        return (len(self.ids) - 1) // self.seq_len

    def __getitem__(self, idx):
        s = idx * self.seq_len
        x = self.ids[s:s+self.seq_len]
        y = self.ids[s+1:s+1+self.seq_len]
        return x, y

def build_loader(text: str, tokenizer: BPETokenizer, seq_len: int, batch_size: int, shuffle=True, num_workers: int=0) -> DataLoader:
    ds = ByteBPETextDataset(text, tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers)
