"""Word-level vocabulary shared by the encoder and the decoder."""

import json
from collections import Counter
from typing import Dict, Iterable, List

from config import (
    EOS_IDX, EOS_TOKEN, PAD_IDX, PAD_TOKEN, SOS_IDX, SOS_TOKEN,
    SPECIAL_TOKENS, UNK_IDX, UNK_TOKEN,
)


class Vocabulary(object):
    """Maps tokens <-> integer ids, with the four special tokens pinned to 0-3."""

    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        # sanity: the special tokens must keep the indices the model assumes
        assert self.stoi[PAD_TOKEN] == PAD_IDX
        assert self.stoi[SOS_TOKEN] == SOS_IDX
        assert self.stoi[EOS_TOKEN] == EOS_IDX
        assert self.stoi[UNK_TOKEN] == UNK_IDX

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    # ------------------------------------------------------------------ #
    # Construction / persistence
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, corpora: Iterable[Iterable[str]], min_freq: int = 2,
              max_size: int = None) -> "Vocabulary":
        counter = Counter()
        for tokens in corpora:
            counter.update(tokens)
        kept = [(tok, c) for tok, c in counter.items() if c >= min_freq]
        # deterministic ordering: frequency desc, then alphabetical
        kept.sort(key=lambda kv: (-kv[1], kv[0]))
        if max_size is not None:
            kept = kept[: max_size - len(SPECIAL_TOKENS)]
        itos = list(SPECIAL_TOKENS) + [tok for tok, _ in kept]
        vocab = cls(itos)
        vocab.freqs = dict(counter)
        return vocab

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"itos": self.itos}, fh, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh)["itos"])

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def encode(self, tokens: List[str], max_len: int,
               add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Token list -> padded id list of exactly ``max_len`` entries."""
        budget = max_len - int(add_sos) - int(add_eos)
        ids = [self.stoi.get(t, UNK_IDX) for t in tokens[:budget]]
        if add_sos:
            ids = [SOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        ids += [PAD_IDX] * (max_len - len(ids))
        return ids

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> List[str]:
        out = []
        for i in ids:
            i = int(i)
            if strip_special:
                if i == EOS_IDX:
                    break
                if i in (PAD_IDX, SOS_IDX):
                    continue
            out.append(self.itos[i])
        return out

    def oov_ratio(self, tokens: List[str]) -> float:
        """Fraction of tokens the vocabulary has never seen (used for OOS gating)."""
        if not tokens:
            return 1.0
        unknown = sum(1 for t in tokens if t not in self.stoi)
        return unknown / float(len(tokens))

    def coverage(self, counter: Dict[str, int]) -> float:
        """Share of running-text tokens covered by the vocabulary."""
        total = sum(counter.values())
        if total == 0:
            return 0.0
        covered = sum(c for tok, c in counter.items() if tok in self.stoi)
        return covered / float(total)
