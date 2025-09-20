# bpe.py
# Minimal, pure-Python BPE for NLP-style subword tokenization (Sennrich et al.)
# - Trains on whitespace-separated words; adds an end-of-word marker '</w>'
# - Greedy apply merges in learned order
# - Not optimized for huge corpora; clear and hackable

from typing import List, Tuple, Dict, Iterable, Optional
from collections import defaultdict
import json
import os

Pair = Tuple[str, str]
Word = Tuple[str, ...]  # e.g., ('l','o','w','</w>')

class BPE:
    def __init__(self, merges: Optional[List[Pair]] = None, *, lowercase: bool = False, eow: str = "</w>"):
        self.lowercase = lowercase
        self.eow = eow
        self.merges: List[Pair] = merges[:] if merges else []
        # Ranks map each pair to its priority (lower is earlier/stronger); useful if you later optimize
        self.ranks: Dict[Pair, int] = {p: i for i, p in enumerate(self.merges)}

    # ---------- Public API ----------
    def train(self, texts: Iterable[str], num_merges: int = 1000, verbose: bool = False) -> None:
        """Learn up to num_merges pair merges from an iterable of text strings."""
        vocab = self._build_vocab(texts)
        merges: List[Pair] = []

        for step in range(num_merges):
            pair_counts = self._count_pairs(vocab)
            if not pair_counts:
                break
            best = max(pair_counts.items(), key=lambda kv: kv[1])[0]
            merges.append(best)
            vocab = self._merge_vocab_once(vocab, best)
            if verbose and (step + 1) % 100 == 0:
                print(f"[BPE] {step+1} merges learned; top pair={best}")

        self.merges = merges
        self.ranks = {p: i for i, p in enumerate(self.merges)}

    def encode(self, text: str) -> List[str]:
        """Tokenize text into BPE subwords (flat list)."""
        if self.lowercase:
            text = text.lower()
        tokens: List[str] = []
        for raw in text.split():
            w: List[str] = list(raw) + [self.eow]
            # Apply merges in learned order
            for pair in self.merges:
                w = self._merge_word_once(w, pair)
            tokens.extend(w)
        return tokens

    def decode(self, tokens: List[str]) -> str:
        """Reconstruct text; uses '</w>' as word boundary."""
        # Join subwords, then turn '</w>' into spaces
        s = "".join(tokens)
        s = s.replace(self.eow, " ")
        return s.strip()

    def save(self, path: str) -> None:
        """Save merges + config as JSON."""
        data = {
            "lowercase": self.lowercase,
            "eow": self.eow,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPE":
        """Load merges + config from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(merges=[tuple(p) for p in data["merges"]],
                   lowercase=bool(data.get("lowercase", False)),
                   eow=data.get("eow", "</w>"))

    # ---------- Training internals ----------
    def _build_vocab(self, texts: Iterable[str]) -> Dict[Word, int]:
        """Build a word->count vocab; each word is tuple of symbols ending with </w>."""
        vocab: Dict[Word, int] = defaultdict(int)
        for line in texts:
            if self.lowercase:
                line = line.lower()
            for w in line.split():
                symbols = tuple(list(w) + [self.eow])
                vocab[symbols] += 1
        return dict(vocab)

    def _count_pairs(self, vocab: Dict[Word, int]) -> Dict[Pair, int]:
        """Count adjacent symbol pairs across vocab, weighted by word freq."""
        counts: Dict[Pair, int] = defaultdict(int)
        for word, freq in vocab.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += freq
        return counts

    def _merge_vocab_once(self, vocab: Dict[Word, int], pair: Pair) -> Dict[Word, int]:
        """Apply one merge to all words in vocab; combine identical outcomes."""
        merged: Dict[Word, int] = defaultdict(int)
        for word, freq in vocab.items():
            new_word = tuple(self._merge_word_once(list(word), pair))
            merged[new_word] += freq
        return dict(merged)

    # ---------- Word-level helpers ----------
    @staticmethod
    def _merge_word_once(word_syms: List[str], pair: Pair) -> List[str]:
        """Merge a given pair once across a word (left-to-right pass)."""
        if len(word_syms) < 2:
            return word_syms
        a, b = pair
        out: List[str] = []
        i = 0
        L = len(word_syms)
        while i < L:
            if i < L - 1 and word_syms[i] == a and word_syms[i + 1] == b:
                out.append(a + b)   # merge
                i += 2
            else:
                out.append(word_syms[i])
                i += 1
        return out


# ---------- Quick demo ----------
if __name__ == "__main__":
    # Tiny corpus (toy example)
    corpus = [
        "low lowest lower",
        "newer wider",
        "low low low",
        "lawn mowers",
    ]

    bpe = BPE(lowercase=True)
    bpe.train(corpus, num_merges=50, verbose=True)

    text = "lower lowest lawn"
    toks = bpe.encode(text)
    print("ENCODED:", toks)
    print("DECODED:", bpe.decode(toks))

    # Save / load
    path = "bpe_merges.json"
    bpe.save(path)
    bpe2 = BPE.load(path)
    print("ENCODED (reloaded):", bpe2.encode(text))
    if os.path.exists(path):
        os.remove(path) 