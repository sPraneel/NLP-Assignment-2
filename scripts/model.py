"""Task 3 - Encoder-Decoder architectures.

Two interchangeable sequence-to-sequence models are provided:

  * ``Seq2SeqTransformer``  - self-attention encoder/decoder (the default).
  * ``Seq2SeqLSTMAttention`` - BiLSTM encoder + LSTM decoder with additive
    (Bahdanau) attention over the encoder states.

Both expose the same three-method interface so that ``train.py``, the decoders
in ``decode.py`` and the Streamlit app are architecture-agnostic:

    logits = model(src, tgt_in)             # teacher forcing, [B, T, V]
    state  = model.init_decode(src)         # encode once
    logits = model.decode_step(state, ys)   # next-token logits, [B, V]
    state  = model.reorder_state(state, ix) # beam bookkeeping
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PAD_IDX


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def make_pad_mask(seq: torch.Tensor) -> torch.Tensor:
    """True where the position is padding. [B, T]"""
    return seq.eq(PAD_IDX)


def causal_mask(size: int, device) -> torch.Tensor:
    """True above the diagonal, i.e. positions the decoder may not look at."""
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)


class PositionalEncoding(nn.Module):
    """Classic fixed sinusoidal positions (Vaswani et al., 2017, Sec. 3.5)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# --------------------------------------------------------------------------- #
# 1. Transformer encoder-decoder
# --------------------------------------------------------------------------- #
class Seq2SeqTransformer(nn.Module):

    arch = "transformer"

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4,
                 n_enc_layers: int = 3, n_dec_layers: int = 3, ffn_dim: int = 512,
                 dropout: float = 0.1, max_len: int = 512):
        super(Seq2SeqTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # One embedding table for both sides: queries and replies share a
        # vocabulary, and tying them cuts parameters by ~1M.
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=n_heads,
            num_encoder_layers=n_enc_layers, num_decoder_layers=n_dec_layers,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.generator = nn.Linear(d_model, vocab_size)
        self.generator.weight = self.embedding.weight   # weight tying
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, seq: torch.Tensor) -> torch.Tensor:
        return self.pos_encoding(self.embedding(seq) * math.sqrt(self.d_model))

    # -- training ---------------------------------------------------------- #
    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_kpm = make_pad_mask(src)
        tgt_kpm = make_pad_mask(tgt_in)
        tgt_mask = causal_mask(tgt_in.size(1), src.device)
        out = self.transformer(
            self._embed(src), self._embed(tgt_in),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        return self.generator(out)

    # -- inference --------------------------------------------------------- #
    def init_decode(self, src: torch.Tensor) -> Dict[str, torch.Tensor]:
        src_kpm = make_pad_mask(src)
        memory = self.transformer.encoder(self._embed(src),
                                          src_key_padding_mask=src_kpm)
        return {"memory": memory, "src_kpm": src_kpm}

    def decode_step(self, state: Dict[str, torch.Tensor],
                    ys: torch.Tensor) -> torch.Tensor:
        """Logits for the token that follows the prefix ``ys``. [B, V]"""
        tgt_mask = causal_mask(ys.size(1), ys.device)
        out = self.transformer.decoder(
            self._embed(ys), state["memory"],
            tgt_mask=tgt_mask,
            memory_key_padding_mask=state["src_kpm"],
        )
        return self.generator(out[:, -1])

    def reorder_state(self, state, index: torch.Tensor):
        return {"memory": state["memory"].index_select(0, index),
                "src_kpm": state["src_kpm"].index_select(0, index)}

    def expand_state(self, state, factor: int):
        return {"memory": state["memory"].repeat_interleave(factor, dim=0),
                "src_kpm": state["src_kpm"].repeat_interleave(factor, dim=0)}


# --------------------------------------------------------------------------- #
# 2. LSTM encoder-decoder with Bahdanau attention
# --------------------------------------------------------------------------- #
class BahdanauAttention(nn.Module):
    """score(h_dec, h_enc) = v^T tanh(W_dec h_dec + W_enc h_enc)."""

    def __init__(self, hidden_dim: int, enc_dim: int):
        super(BahdanauAttention, self).__init__()
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_enc = nn.Linear(enc_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_out, src_pad_mask):
        # dec_hidden [B, H] ; enc_out [B, S, E] ; src_pad_mask [B, S]
        scores = self.v(torch.tanh(
            self.W_dec(dec_hidden).unsqueeze(1) + self.W_enc(enc_out)
        )).squeeze(-1)                                  # [B, S]
        scores = scores.masked_fill(src_pad_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)             # [B, S]
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)   # [B, E]
        return context, weights


class Seq2SeqLSTMAttention(nn.Module):

    arch = "lstm_attn"

    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 512,
                 n_layers: int = 1, dropout: float = 0.1):
        super(Seq2SeqLSTMAttention, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        enc_dim = hidden_dim * 2
        self.bridge_h = nn.Linear(enc_dim, hidden_dim)
        self.bridge_c = nn.Linear(enc_dim, hidden_dim)

        self.attention = BahdanauAttention(hidden_dim, enc_dim)
        # input-feeding: previous context is concatenated to the next embedding
        self.decoder = nn.LSTM(emb_dim + enc_dim, hidden_dim, num_layers=n_layers,
                               batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden_dim + enc_dim, emb_dim)
        self.generator = nn.Linear(emb_dim, vocab_size)
        self.generator.weight = self.embedding.weight   # weight tying

    # -- encoder ----------------------------------------------------------- #
    def _encode(self, src):
        emb = self.dropout(self.embedding(src))
        lengths = (~make_pad_mask(src)).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        enc_out, (h, c) = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(
            enc_out, batch_first=True, total_length=src.size(1))
        # merge the two directions of every layer -> decoder initial state
        h = h.view(self.n_layers, 2, h.size(1), self.hidden_dim)
        c = c.view(self.n_layers, 2, c.size(1), self.hidden_dim)
        h0 = torch.tanh(self.bridge_h(torch.cat([h[:, 0], h[:, 1]], dim=-1)))
        c0 = torch.tanh(self.bridge_c(torch.cat([c[:, 0], c[:, 1]], dim=-1)))
        return enc_out, (h0.contiguous(), c0.contiguous())

    def _step(self, token, hidden, context, enc_out, src_pad_mask):
        """One decoder time step with input feeding."""
        emb = self.dropout(self.embedding(token))               # [B, 1, E]
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, hidden = self.decoder(rnn_in, hidden)              # [B, 1, H]
        dec_h = out.squeeze(1)
        context, attn = self.attention(dec_h, enc_out, src_pad_mask)
        feat = torch.tanh(self.out_proj(torch.cat([dec_h, context], dim=-1)))
        logits = self.generator(self.dropout(feat))             # [B, V]
        return logits, hidden, context, attn

    # -- training ---------------------------------------------------------- #
    def forward(self, src, tgt_in):
        enc_out, hidden = self._encode(src)
        src_pad_mask = make_pad_mask(src)
        context = enc_out.new_zeros(src.size(0), enc_out.size(-1))
        logits = []
        for t in range(tgt_in.size(1)):
            step_logits, hidden, context, _ = self._step(
                tgt_in[:, t: t + 1], hidden, context, enc_out, src_pad_mask)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)                       # [B, T, V]

    # -- inference --------------------------------------------------------- #
    def init_decode(self, src):
        enc_out, hidden = self._encode(src)
        return {
            "enc_out": enc_out,
            "src_kpm": make_pad_mask(src),
            "hidden": hidden,
            "context": enc_out.new_zeros(src.size(0), enc_out.size(-1)),
            "attn": None,
        }

    def decode_step(self, state, ys):
        logits, hidden, context, attn = self._step(
            ys[:, -1:], state["hidden"], state["context"],
            state["enc_out"], state["src_kpm"])
        state["hidden"], state["context"], state["attn"] = hidden, context, attn
        return logits

    def reorder_state(self, state, index):
        h, c = state["hidden"]
        return {
            "enc_out": state["enc_out"].index_select(0, index),
            "src_kpm": state["src_kpm"].index_select(0, index),
            "hidden": (h.index_select(1, index), c.index_select(1, index)),
            "context": state["context"].index_select(0, index),
            "attn": None if state["attn"] is None else state["attn"].index_select(0, index),
        }

    def expand_state(self, state, factor):
        h, c = state["hidden"]
        return {
            "enc_out": state["enc_out"].repeat_interleave(factor, dim=0),
            "src_kpm": state["src_kpm"].repeat_interleave(factor, dim=0),
            "hidden": (h.repeat_interleave(factor, dim=1),
                       c.repeat_interleave(factor, dim=1)),
            "context": state["context"].repeat_interleave(factor, dim=0),
            "attn": None,
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(arch: str, vocab_size: int, cfg) -> nn.Module:
    if arch == "transformer":
        return Seq2SeqTransformer(
            vocab_size, d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
            n_enc_layers=cfg.N_ENC_LAYERS, n_dec_layers=cfg.N_DEC_LAYERS,
            ffn_dim=cfg.FFN_DIM, dropout=cfg.DROPOUT,
            max_len=max(cfg.MAX_SRC_LEN, cfg.MAX_TGT_LEN) + 8,
        )
    if arch == "lstm_attn":
        return Seq2SeqLSTMAttention(
            vocab_size, emb_dim=cfg.EMB_DIM, hidden_dim=cfg.HIDDEN_DIM,
            n_layers=cfg.LSTM_LAYERS, dropout=cfg.DROPOUT,
        )
    raise ValueError("unknown architecture: {}".format(arch))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
