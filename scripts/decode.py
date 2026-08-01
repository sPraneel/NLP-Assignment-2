"""Task 3.3 - Greedy and beam-search decoding, plus the out-of-scope gate.

``ResponseGenerator`` is the single object the Streamlit application and the
evaluation script both use, so what an evaluator measures is exactly what a user
sees.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from data import pick_device
from model import build_model
from preprocess import content_words, detokenize, preprocess_query
from vocab import Vocabulary


# --------------------------------------------------------------------------- #
# Repetition control
# --------------------------------------------------------------------------- #
def banned_by_no_repeat_ngram(sequence: List[int], n: int) -> List[int]:
    """Tokens that would complete an n-gram already present in ``sequence``."""
    if n <= 0 or len(sequence) < n:
        return []
    prefix = tuple(sequence[-(n - 1):]) if n > 1 else ()
    banned = []
    for i in range(len(sequence) - n + 1):
        gram = tuple(sequence[i: i + n])
        if gram[:-1] == prefix:
            banned.append(gram[-1])
    return banned


# --------------------------------------------------------------------------- #
# Decoders
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, max_len: int = C.MAX_DECODE_LEN,
                  no_repeat_ngram: int = C.NO_REPEAT_NGRAM
                  ) -> Tuple[List[List[int]], List[float]]:
    """Argmax decoding for a batch of queries. Returns (token ids, mean log-prob)."""
    device = src.device
    batch = src.size(0)
    state = model.init_decode(src)
    ys = torch.full((batch, 1), C.SOS_IDX, dtype=torch.long, device=device)

    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    logprob_sum = torch.zeros(batch, device=device)
    lengths = torch.zeros(batch, device=device)

    for _ in range(max_len - 1):
        logits = model.decode_step(state, ys)                 # [B, V]
        logits[:, C.PAD_IDX] = float("-inf")
        logits[:, C.UNK_IDX] = float("-inf")
        logits[:, C.SOS_IDX] = float("-inf")
        if no_repeat_ngram:
            for b in range(batch):
                for tok in banned_by_no_repeat_ngram(ys[b].tolist(), no_repeat_ngram):
                    logits[b, tok] = float("-inf")

        logprobs = F.log_softmax(logits, dim=-1)
        best_lp, best = logprobs.max(dim=-1)

        active = ~finished
        logprob_sum += best_lp * active
        lengths += active.float()
        best = torch.where(finished, torch.full_like(best, C.PAD_IDX), best)
        ys = torch.cat([ys, best.unsqueeze(1)], dim=1)
        finished = finished | best.eq(C.EOS_IDX)
        if bool(finished.all()):
            break

    outputs, scores = [], []
    for b in range(batch):
        seq = [int(t) for t in ys[b, 1:].tolist()]
        if C.EOS_IDX in seq:
            seq = seq[: seq.index(C.EOS_IDX)]
        outputs.append(seq)
        scores.append(float(logprob_sum[b] / max(float(lengths[b]), 1.0)))
    return outputs, scores


@torch.no_grad()
def beam_search_decode(model, src: torch.Tensor, beam_size: int = C.BEAM_SIZE,
                       max_len: int = C.MAX_DECODE_LEN,
                       length_penalty: float = C.LENGTH_PENALTY,
                       no_repeat_ngram: int = C.NO_REPEAT_NGRAM
                       ) -> Tuple[List[int], float]:
    """Beam search for a **single** query (src is [1, S]).

    Hypotheses are ranked by ``logprob_sum / length**alpha`` (Wu et al., 2016),
    which stops the short generic replies from always winning.
    """
    device = src.device
    state = model.expand_state(model.init_decode(src), beam_size)
    ys = torch.full((beam_size, 1), C.SOS_IDX, dtype=torch.long, device=device)

    scores = torch.full((beam_size,), float("-inf"), device=device)
    scores[0] = 0.0                       # only the first beam is live at t=0
    finished: List[Tuple[float, List[int], int]] = []

    for step in range(max_len - 1):
        logits = model.decode_step(state, ys)
        logits[:, C.PAD_IDX] = float("-inf")
        logits[:, C.UNK_IDX] = float("-inf")
        logits[:, C.SOS_IDX] = float("-inf")
        if no_repeat_ngram:
            for b in range(ys.size(0)):
                for tok in banned_by_no_repeat_ngram(ys[b].tolist(), no_repeat_ngram):
                    logits[b, tok] = float("-inf")

        logprobs = F.log_softmax(logits, dim=-1)
        cand = scores.unsqueeze(1) + logprobs                 # [beam, V]
        flat = cand.view(-1)
        top_scores, top_ix = flat.topk(beam_size)
        beam_ix = torch.div(top_ix, logprobs.size(-1), rounding_mode="floor")
        token_ix = top_ix % logprobs.size(-1)

        ys = torch.cat([ys.index_select(0, beam_ix), token_ix.unsqueeze(1)], dim=1)
        state = model.reorder_state(state, beam_ix)
        scores = top_scores

        # Retire any beam that produced <eos>; keep the slot alive but dead.
        for b in range(beam_size):
            if int(token_ix[b]) == C.EOS_IDX:
                seq = [int(t) for t in ys[b, 1:-1].tolist()]
                finished.append((float(scores[b]), seq, len(seq) + 1))
                scores[b] = float("-inf")
        if len(finished) >= beam_size or bool(torch.isinf(scores).all()):
            break

    if not finished:      # hit max_len without any beam emitting <eos>
        for b in range(beam_size):
            if not torch.isinf(scores[b]):
                seq = [int(t) for t in ys[b, 1:].tolist()]
                finished.append((float(scores[b]), seq, len(seq)))

    def normalised(item):
        total, seq, length = item
        return total / (max(length, 1) ** length_penalty)

    best_total, best_seq, best_len = max(finished, key=normalised)
    return best_seq, best_total / max(best_len, 1)            # mean log-prob


# --------------------------------------------------------------------------- #
# End-to-end generator used by the application
# --------------------------------------------------------------------------- #
class ResponseGenerator(object):
    """Query text -> support reply, with scope checking and readable output."""

    def __init__(self, checkpoint: str = C.CHECKPOINT,
                 vocab_path: str = C.VOCAB_JSON,
                 placeholder_map: Optional[str] = None,
                 scope_lexicon: Optional[str] = None,
                 device: str = "auto"):
        self.device = pick_device(device)
        self.vocab = Vocabulary.load(vocab_path)

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.train_config = ckpt.get("config", {})
        arch = self.train_config.get("arch", C.ARCH)
        self.model = build_model(arch, len(self.vocab), C).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        with open(placeholder_map or C.PLACEHOLDER_JSON, "r", encoding="utf-8") as fh:
            self.placeholder_map = json.load(fh)

        with open(scope_lexicon or C.SCOPE_LEXICON_JSON, "r", encoding="utf-8") as fh:
            lexicon = json.load(fh)
        self.query_vocab = set(lexicon["query_vocab"])
        self.anchors = set(lexicon["anchors"])

    # ------------------------------------------------------------------ #
    def _encode_query(self, text: str):
        tokens = preprocess_query(text, self.placeholder_map)
        ids = self.vocab.encode(tokens, C.MAX_SRC_LEN, add_sos=True, add_eos=True)
        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        return tokens, src

    def _scope_check(self, tokens: List[str], avg_logprob: float) -> Dict:
        """Two cheap, explainable signals decide whether we answer at all.

        1. the share of *content* words the training queries have never seen;
        2. the decoder's own mean per-token log-probability.
        """
        content = content_words(tokens)
        if content:
            unknown = [t for t in content if t not in self.query_vocab]
            oov = len(unknown) / float(len(content))
        else:
            # only stopwords and punctuation: nothing to ground an answer on,
            # unless the query still contains a known slot such as an order number
            unknown, oov = [], (0.0 if any(t.startswith("<ph_") for t in tokens) else 1.0)

        n_anchors = sum(1 for t in content if t in self.anchors)
        in_scope = (oov <= C.OOS_OOV_RATIO) and (avg_logprob >= C.OOS_MIN_AVG_LOGPROB)

        if oov > C.OOS_OOV_RATIO:
            reason = ("{:.0%} of the meaningful words in this query never appear in "
                      "the support corpus ({})".format(oov, ", ".join(unknown[:5]) or "none"))
        elif avg_logprob < C.OOS_MIN_AVG_LOGPROB:
            reason = ("the decoder is not confident (mean log-probability "
                      "{:.2f})".format(avg_logprob))
        else:
            reason = "in scope ({} known support term{})".format(
                n_anchors, "" if n_anchors == 1 else "s")

        return {"in_scope": in_scope, "oov_ratio": round(oov, 3),
                "n_anchors": n_anchors, "unknown_words": unknown[:8],
                "avg_logprob": round(avg_logprob, 3), "reason": reason}

    # ------------------------------------------------------------------ #
    def generate(self, text: str, strategy: str = C.DECODE_STRATEGY,
                 beam_size: int = C.BEAM_SIZE, max_len: int = C.MAX_DECODE_LEN,
                 apply_scope_check: bool = True) -> Dict:
        """Returns the reply plus every diagnostic the report and UI display."""
        tokens, src = self._encode_query(text)
        if not tokens:
            return {"query": text, "response": C.OOS_MESSAGE, "tokens": [],
                    "raw_tokens": [], "strategy": strategy, "in_scope": False,
                    "oov_ratio": 1.0, "avg_logprob": 0.0,
                    "reason": "empty query after cleaning"}

        if strategy == "beam":
            ids, avg_lp = beam_search_decode(self.model, src, beam_size=beam_size,
                                             max_len=max_len)
        else:
            batch_ids, batch_lp = greedy_decode(self.model, src, max_len=max_len)
            ids, avg_lp = batch_ids[0], batch_lp[0]

        out_tokens = self.vocab.decode(ids)
        scope = self._scope_check(tokens, avg_lp)
        response = (detokenize(out_tokens, self.placeholder_map)
                    if (scope["in_scope"] or not apply_scope_check) else C.OOS_MESSAGE)

        result = {"query": text, "query_tokens": tokens, "response": response,
                  "raw_tokens": out_tokens, "strategy": strategy,
                  "beam_size": beam_size if strategy == "beam" else None}
        result.update(scope)
        return result

    def generate_batch(self, texts: List[str], strategy: str = "greedy",
                       max_len: int = C.MAX_DECODE_LEN,
                       apply_scope_check: bool = True,
                       batch_size: int = 32) -> List[Dict]:
        """Document upload path - greedy decoding batches, beam falls back to a loop."""
        if strategy != "greedy":
            return [self.generate(t, strategy=strategy, max_len=max_len,
                                  apply_scope_check=apply_scope_check) for t in texts]

        results = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start: start + batch_size]
            token_lists, rows = [], []
            for text in chunk:
                toks, _ = self._encode_query(text)
                token_lists.append(toks)
                rows.append(self.vocab.encode(toks, C.MAX_SRC_LEN,
                                              add_sos=True, add_eos=True))
            src = torch.tensor(rows, dtype=torch.long, device=self.device)
            ids_batch, lp_batch = greedy_decode(self.model, src, max_len=max_len)

            for text, toks, ids, lp in zip(chunk, token_lists, ids_batch, lp_batch):
                out_tokens = self.vocab.decode(ids)
                scope = self._scope_check(toks, lp)
                response = (detokenize(out_tokens, self.placeholder_map)
                            if (scope["in_scope"] or not apply_scope_check) else C.OOS_MESSAGE)
                row = {"query": text, "query_tokens": toks, "response": response,
                       "raw_tokens": out_tokens, "strategy": "greedy", "beam_size": None}
                row.update(scope)
                results.append(row)
        return results


# --------------------------------------------------------------------------- #
def main():
    import argparse

    p = argparse.ArgumentParser(description="Generate replies from the command line")
    p.add_argument("queries", nargs="*", help="one or more customer queries")
    p.add_argument("--checkpoint", default=C.CHECKPOINT)
    p.add_argument("--strategy", default="beam", choices=["greedy", "beam"])
    p.add_argument("--beam-size", type=int, default=C.BEAM_SIZE)
    p.add_argument("--no-scope-check", action="store_true")
    args = p.parse_args()

    demo = args.queries or [
        "i want to cancel order 4471902",
        "how can i get a refund for my last purchase?",
        "i cannot log into my account, i forgot my password",
        "where is my package? it has not arrived yet",
        "i need to speak to a human agent",
        "what is the boiling point of water on mars",     # out of domain
    ]
    gen = ResponseGenerator(checkpoint=args.checkpoint)
    for q in demo:
        r = gen.generate(q, strategy=args.strategy, beam_size=args.beam_size,
                         apply_scope_check=not args.no_scope_check)
        print("\nQ: {}".format(q))
        print("A: {}".format(r["response"]))
        print("   [in_scope={} oov={} avg_logprob={} | {}]".format(
            r["in_scope"], r["oov_ratio"], r["avg_logprob"], r["reason"]))


if __name__ == "__main__":
    main()
