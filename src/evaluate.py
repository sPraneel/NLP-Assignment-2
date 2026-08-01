"""Task 5.1 - Automatic evaluation of the generated support replies.

Reports, on the held-out test split:

  * corpus BLEU and sentence-BLEU (sacreBLEU)
  * ROUGE-1 / ROUGE-2 / ROUGE-L F1 (Google ``rouge_score``)
  * token-level perplexity of the reference replies under the model
  * distinct-1/2 and a generic-reply rate, which expose the "safe answer"
    failure mode that seq2seq chatbots are known for
  * the out-of-scope gate's behaviour on in-domain vs out-of-domain queries

It also writes ``reports/manual_rating_sheet.csv``, a random sample for the
manual relevance rating the assignment asks for.

Run:  python src/evaluate.py --limit 400
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from data import SupportPairDataset, collate, pick_device
from decode import ResponseGenerator
from preprocess import detokenize
from torch.utils.data import DataLoader

# Queries that are deliberately outside the support domain (Task 5.2).
OUT_OF_DOMAIN_QUERIES = [
    "what is the boiling point of water on mars",
    "write me a python function that sorts a list",
    "who won the football world cup in 1998",
    "can you tell me a joke about penguins",
    "what should i cook for dinner tonight",
    "explain the theory of general relativity",
]


# --------------------------------------------------------------------------- #
# Perplexity
# --------------------------------------------------------------------------- #
@torch.no_grad()
def corpus_perplexity(model, csv_path: str, vocab, device, batch_size: int = 32) -> float:
    """exp(mean token cross-entropy) of the *reference* replies."""
    ds = SupportPairDataset(csv_path, vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    criterion = nn.CrossEntropyLoss(ignore_index=C.PAD_IDX, reduction="sum")
    total_loss, total_tokens = 0.0, 0
    model.eval()
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        total_loss += float(criterion(logits.reshape(-1, logits.size(-1)),
                                      tgt_out.reshape(-1)))
        total_tokens += int((tgt_out != C.PAD_IDX).sum())
    return math.exp(total_loss / max(total_tokens, 1))


# --------------------------------------------------------------------------- #
# Diversity / genericity
# --------------------------------------------------------------------------- #
def distinct_n(token_lists: List[List[str]], n: int) -> float:
    grams = Counter()
    for toks in token_lists:
        for i in range(len(toks) - n + 1):
            grams[tuple(toks[i: i + n])] += 1
    total = sum(grams.values())
    return len(grams) / float(total) if total else 0.0


def generic_rate(responses: List[str], top_k: int = 5) -> Dict:
    """How often the model falls back on its handful of favourite replies."""
    counts = Counter(r.strip().lower() for r in responses)
    top = counts.most_common(top_k)
    repeated = sum(c for _, c in counts.items() if c > 1)
    return {
        "unique_responses": len(counts),
        "repeated_response_rate": round(repeated / max(len(responses), 1), 4),
        "most_common": [{"count": c, "response": r[:160]} for r, c in top],
    }


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Evaluate the response generator")
    p.add_argument("--checkpoint", default=C.CHECKPOINT)
    p.add_argument("--split", default=C.TEST_CSV)
    p.add_argument("--limit", type=int, default=400,
                   help="number of test queries to decode (decoding is the slow part)")
    p.add_argument("--strategy", default="beam", choices=["greedy", "beam", "both"])
    p.add_argument("--beam-size", type=int, default=C.BEAM_SIZE)
    p.add_argument("--sample-size", type=int, default=20,
                   help="rows written to the manual rating sheet")
    p.add_argument("--out", default=C.METRICS_JSON)
    args = p.parse_args()

    import sacrebleu
    from rouge_score import rouge_scorer

    random.seed(C.RANDOM_SEED)
    os.makedirs(C.REPORT_DIR, exist_ok=True)

    print("Loading model from {}".format(os.path.relpath(args.checkpoint, C.ROOT_DIR)))
    gen = ResponseGenerator(checkpoint=args.checkpoint)
    device = gen.device

    test_df = pd.read_csv(args.split).fillna("")
    if args.limit and args.limit < len(test_df):
        test_df = test_df.sample(n=args.limit, random_state=C.RANDOM_SEED)
    test_df = test_df.reset_index(drop=True)
    queries = [str(q) for q in test_df["query_clean"]]
    references = [str(r) for r in test_df["response_clean"]]
    print("Evaluating on {} held-out queries".format(len(queries)))

    metrics = {
        "checkpoint": os.path.basename(args.checkpoint),
        "arch": gen.train_config.get("arch"),
        "n_test_queries": len(queries),
        "train_config": gen.train_config,
    }

    # ---- perplexity over the FULL test split ---------------------------- #
    print("[1/4] perplexity on the full test split ...")
    metrics["perplexity_test"] = round(
        corpus_perplexity(gen.model, args.split, gen.vocab, device), 3)
    metrics["perplexity_valid"] = round(
        corpus_perplexity(gen.model, C.VALID_CSV, gen.vocab, device), 3)
    print("      test perplexity {}".format(metrics["perplexity_test"]))

    # ---- decode and score ------------------------------------------------ #
    strategies = ["greedy", "beam"] if args.strategy == "both" else [args.strategy]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_strategy = {}
    decoded_cache = {}

    for si, strategy in enumerate(strategies, start=2):
        print("[{}/4] decoding with {} ...".format(si, strategy))
        if strategy == "greedy":
            results = gen.generate_batch(queries, strategy="greedy",
                                         apply_scope_check=False)
        else:
            results = []
            for i, q in enumerate(queries):
                results.append(gen.generate(q, strategy="beam",
                                            beam_size=args.beam_size,
                                            apply_scope_check=False))
                if (i + 1) % 50 == 0:
                    print("      {}/{}".format(i + 1, len(queries)), flush=True)

        hypotheses = [" ".join(r["raw_tokens"]) for r in results]
        decoded_cache[strategy] = results

        bleu = sacrebleu.corpus_bleu(hypotheses, [references],
                                     tokenize="13a", lowercase=True)
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        sent_bleu = [sacrebleu.sentence_bleu(h, [r]).score
                     for h, r in zip(hypotheses, references)]

        rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for hyp, ref in zip(hypotheses, references):
            sc = scorer.score(ref, hyp)
            for k in rouge_totals:
                rouge_totals[k] += sc[k].fmeasure
        n = max(len(hypotheses), 1)

        hyp_tokens = [h.split() for h in hypotheses]
        per_strategy[strategy] = {
            "bleu_corpus": round(bleu.score, 3),
            "bleu_precisions": [round(x, 2) for x in bleu.precisions],
            "bleu_sentence_mean": round(sum(sent_bleu) / n, 3),
            "chrf": round(chrf.score, 3),
            "rouge1_f": round(rouge_totals["rouge1"] / n, 4),
            "rouge2_f": round(rouge_totals["rouge2"] / n, 4),
            "rougeL_f": round(rouge_totals["rougeL"] / n, 4),
            "distinct_1": round(distinct_n(hyp_tokens, 1), 4),
            "distinct_2": round(distinct_n(hyp_tokens, 2), 4),
            "mean_hypothesis_tokens": round(sum(len(t) for t in hyp_tokens) / n, 2),
            "mean_reference_tokens": round(
                sum(len(r.split()) for r in references) / n, 2),
            "genericity": generic_rate(hypotheses),
        }
        print("      BLEU {:.2f} | ROUGE-L {:.4f} | distinct-2 {:.4f}".format(
            per_strategy[strategy]["bleu_corpus"],
            per_strategy[strategy]["rougeL_f"],
            per_strategy[strategy]["distinct_2"]))

    metrics["by_strategy"] = per_strategy

    # ---- out-of-scope gate ---------------------------------------------- #
    print("[4/4] out-of-scope gate ...")
    in_domain = gen.generate_batch(queries[:200], strategy="greedy",
                                   apply_scope_check=True)
    ood = [gen.generate(q, strategy="beam", apply_scope_check=True)
           for q in OUT_OF_DOMAIN_QUERIES]
    metrics["scope_gate"] = {
        "in_domain_accepted": round(
            sum(1 for r in in_domain if r["in_scope"]) / max(len(in_domain), 1), 4),
        "out_of_domain_rejected": round(
            sum(1 for r in ood if not r["in_scope"]) / max(len(ood), 1), 4),
        "thresholds": {"oov_ratio": C.OOS_OOV_RATIO,
                       "min_avg_logprob": C.OOS_MIN_AVG_LOGPROB},
        "out_of_domain_detail": [
            {"query": r["query"], "in_scope": r["in_scope"],
             "oov_ratio": r["oov_ratio"], "avg_logprob": r["avg_logprob"],
             "reason": r["reason"]} for r in ood],
    }
    print("      in-domain accepted {:.1%} | out-of-domain rejected {:.1%}".format(
        metrics["scope_gate"]["in_domain_accepted"],
        metrics["scope_gate"]["out_of_domain_rejected"]))

    # ---- manual rating sheet -------------------------------------------- #
    primary = strategies[-1]
    idx = random.sample(range(len(queries)), min(args.sample_size, len(queries)))
    rows = []
    for i in idx:
        r = decoded_cache[primary][i]
        rows.append({
            "query": detokenize(queries[i].split(), gen.placeholder_map),
            "reference_reply": detokenize(references[i].split(), gen.placeholder_map),
            "generated_reply": detokenize(r["raw_tokens"], gen.placeholder_map),
            "intent": test_df.get("intent", pd.Series([""] * len(test_df)))[i],
            "relevance_1_to_5": "",
            "fluency_1_to_5": "",
            "would_send_as_is_yes_no": "",
            "rater_comment": "",
        })
    sheet = os.path.join(C.REPORT_DIR, "manual_rating_sheet.csv")
    pd.DataFrame(rows).to_csv(sheet, index=False)

    # ---- qualitative sample --------------------------------------------- #
    samples = []
    for i in idx[:8]:
        r = decoded_cache[primary][i]
        samples.append({
            "query": detokenize(queries[i].split(), gen.placeholder_map),
            "reference": detokenize(references[i].split(), gen.placeholder_map),
            "generated": detokenize(r["raw_tokens"], gen.placeholder_map),
        })
    metrics["qualitative_samples"] = samples

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n" + "=" * 78)
    print("RESULTS  (architecture: {}, {} test queries)".format(
        metrics["arch"], metrics["n_test_queries"]))
    print("=" * 78)
    print("  perplexity (valid / test) : {} / {}".format(
        metrics["perplexity_valid"], metrics["perplexity_test"]))
    for strategy, m in per_strategy.items():
        print("  --- {} decoding".format(strategy))
        print("      BLEU              : {}".format(m["bleu_corpus"]))
        print("      chrF              : {}".format(m["chrf"]))
        print("      ROUGE-1 / 2 / L   : {} / {} / {}".format(
            m["rouge1_f"], m["rouge2_f"], m["rougeL_f"]))
        print("      distinct-1 / -2   : {} / {}".format(m["distinct_1"], m["distinct_2"]))
        print("      unique replies    : {}/{}".format(
            m["genericity"]["unique_responses"], metrics["n_test_queries"]))
        print("      mean length hyp/ref: {} / {} tokens".format(
            m["mean_hypothesis_tokens"], m["mean_reference_tokens"]))
    print("  scope gate: {:.1%} in-domain accepted, {:.1%} out-of-domain rejected".format(
        metrics["scope_gate"]["in_domain_accepted"],
        metrics["scope_gate"]["out_of_domain_rejected"]))
    print("\nMetrics      -> {}".format(os.path.relpath(args.out, C.ROOT_DIR)))
    print("Rating sheet -> {}".format(os.path.relpath(sheet, C.ROOT_DIR)))


if __name__ == "__main__":
    main()
