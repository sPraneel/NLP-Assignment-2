"""Torch ``Dataset``/``DataLoader`` plumbing for the processed splits."""

import os
import sys
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from vocab import Vocabulary


class SupportPairDataset(Dataset):
    """Query/response pairs already cleaned and tokenised by ``preprocess.py``."""

    def __init__(self, csv_path: str, vocab: Vocabulary,
                 max_src_len: int = C.MAX_SRC_LEN,
                 max_tgt_len: int = C.MAX_TGT_LEN):
        self.df = pd.read_csv(csv_path).fillna("")
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.queries = [str(s).split() for s in self.df["query_clean"]]
        self.responses = [str(s).split() for s in self.df["response_clean"]]
        self.categories = list(self.df["category"]) if "category" in self.df else [""] * len(self.df)
        self.intents = list(self.df["intent"]) if "intent" in self.df else [""] * len(self.df)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.vocab.encode(self.queries[idx], self.max_src_len,
                                add_sos=True, add_eos=True)
        tgt = self.vocab.encode(self.responses[idx], self.max_tgt_len,
                                add_sos=True, add_eos=True)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

    def raw_pair(self, idx: int):
        return self.queries[idx], self.responses[idx]


def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Stacks a batch and trims the padding that no example in it needs.

    Sequences are pre-padded to a fixed length, but most batches are far shorter
    than the maximum; trimming to the longest member is a free ~2x speed-up.
    """
    src = torch.stack([b[0] for b in batch])
    tgt = torch.stack([b[1] for b in batch])
    src_len = int((src != C.PAD_IDX).sum(dim=1).max())
    tgt_len = int((tgt != C.PAD_IDX).sum(dim=1).max())
    src = src[:, :src_len]
    tgt = tgt[:, :tgt_len]
    # teacher forcing: the decoder reads <sos> w1 .. wn-1 and predicts w1 .. <eos>
    return src, tgt[:, :-1].contiguous(), tgt[:, 1:].contiguous()


def make_loaders(vocab: Vocabulary, batch_size: int = C.BATCH_SIZE,
                 num_workers: int = 0):
    loaders = {}
    for name, path in (("train", C.TRAIN_CSV), ("valid", C.VALID_CSV), ("test", C.TEST_CSV)):
        ds = SupportPairDataset(path, vocab)
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            collate_fn=collate, num_workers=num_workers, drop_last=False,
        )
    return loaders


def pick_device(prefer: str = "auto") -> torch.device:
    """MPS on Apple silicon, CUDA where available, CPU otherwise."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
