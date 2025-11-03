from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import json

@dataclass
class BPETokenizerConfig:
    vocab_size: int = 1024  # includes 256 bytes

class BPETokenizer:
    """Minimal byte-level BPE tokenizer (train + encode/decode).
    - Base alphabet: tokens 0..255 (bytes)
    - Greedy merges by learned pair rank
    """
    def __init__(self, merges: List[Tuple[int,int]]|None=None, vocab_size: int|None=None):
        self.base = 256
        self.merges: List[Tuple[int,int]] = merges or []
        self.vocab_size = vocab_size if vocab_size is not None else self.base + len(self.merges)
        self._build_maps()

    def _build_maps(self):
        self.tok2bytes: Dict[int, List[int]] = {i: [i] for i in range(256)}
        tid = 256
        for a,b in self.merges:
            self.tok2bytes[tid] = self.tok2bytes[a] + self.tok2bytes[b]
            tid += 1
        self.rank = {pair:i for i,pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

    # ---------- training ----------
    @staticmethod
    def _split_words_bytes(text: bytes) -> List[List[int]]:
        words: List[List[int]] = []
        cur: List[int] = []
        for b in text:
            if b in (9,10,11,12,13,32):
                if cur: words.append(cur); cur = []
                words.append([b])
            else:
                cur.append(b)
        if cur: words.append(cur)
        return words

    @staticmethod
    def _pair_stats(seqs: List[List[int]]):
        stats = {}
        for s in seqs:
            for i in range(len(s)-1):
                p=(s[i], s[i+1])
                stats[p]=stats.get(p,0)+1
        return stats

    @staticmethod
    def _merge_pair(seqs: List[List[int]], pair: Tuple[int,int], new_id: int) -> List[List[int]]:
        a,b = pair
        out: List[List[int]] = []
        for s in seqs:
            i=0; m=[]
            while i < len(s):
                if i<len(s)-1 and s[i]==a and s[i+1]==b:
                    m.append(new_id); i+=2
                else:
                    m.append(s[i]); i+=1
            out.append(m)
        return out

    @classmethod
    def train(cls, texts: List[str], vocab_size: int = 1024, progress: bool=True) -> "BPETokenizer":
        assert vocab_size >= 256
        seqs: List[List[int]] = []
        for t in texts: seqs.extend(cls._split_words_bytes(t.encode("utf-8")))
        merges: List[Tuple[int,int]] = []
        cur = 256
        while cur < vocab_size:
            stats = cls._pair_stats(seqs)
            if not stats: break
            (a,b),freq = max(stats.items(), key=lambda kv: kv[1])
            seqs = cls._merge_pair(seqs, (a,b), cur)
            merges.append((a,b)); cur += 1
            if progress and len(merges) % 100 == 0:
                print(f"[BPE] merges={len(merges)} freq={freq}")
        return cls(merges=merges, vocab_size=cur)

    # ---------- encode/decode ----------
    def _word_to_tokens(self, word: List[int]) -> List[int]:
        if len(word) < 2 or not self.rank: return word
        syms = word[:]
        while True:
            best = None; best_rank = None
            for i in range(len(syms)-1):
                p=(syms[i], syms[i+1])
                if p in self.rank:
                    r = self.rank[p]
                    if best_rank is None or r < best_rank:
                        best, best_rank = p, r
            if best is None: break
            a,b = best
            i=0; m=[]
            while i < len(syms):
                if i<len(syms)-1 and syms[i]==a and syms[i+1]==b:
                    m.append(256 + self.rank[(a,b)]); i+=2
                else:
                    m.append(syms[i]); i+=1
            syms = m
        return syms

    def encode(self, text: str) -> List[int]:
        toks: List[int] = []
        for w in self._split_words_bytes(text.encode("utf-8")):
            toks.extend(self._word_to_tokens(w))
        return toks

    def decode(self, ids: List[int]) -> str:
        out: List[int] = []
        for tid in ids:
            if tid in self.tok2bytes: out.extend(self.tok2bytes[tid])
        return bytes(bytearray(out)).decode("utf-8", errors="replace")

    # ---------- persistence ----------
    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "merges": self.merges}, f)

    @classmethod
    def from_json(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(merges=[tuple(x) for x in obj["merges"]], vocab_size=obj["vocab_size"])
