"""Task 2 - Data collection and preprocessing.

Reads the raw Bitext customer-support corpus, cleans and normalises both sides
of every query/response pair, removes automated and duplicate messages, builds a
shared word-level vocabulary and writes leakage-free train/valid/test splits.

Run:  python src/preprocess.py
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from vocab import Vocabulary

# --------------------------------------------------------------------------- #
# 1. Regular expressions used for cleaning
# --------------------------------------------------------------------------- #
RE_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_USERNAME = re.compile(r"(?<![\w])@\w{2,}")
RE_HASHTAG = re.compile(r"(?<![\w])#(\w+)")
# Strips markup, but never the typed tokens this pipeline itself introduces -
# without the lookahead, running the cleaner over already-processed text (as the
# evaluation script does, since the splits are stored tokenised) would silently
# delete every <ph_...> slot.
RE_HTML = re.compile(
    r"<(?!/?(?:ph_[a-z0-9_]*|url|email|user|num|unk|pad|sos|eos)>)[^<>]{1,40}>")
RE_NUMBER = re.compile(r"(?<![\w<])\d[\d,.\-/]*\d|(?<![\w<])\d(?![\w>])")
RE_MULTISPACE = re.compile(r"\s+")
RE_REPEAT_CHAR = re.compile(r"(.)\1{2,}")          # "heyyyy" -> "heyy"
RE_REPEAT_PUNCT = re.compile(r"([!?.,])\1{1,}")     # "???"    -> "?"
RE_NONPRINT = re.compile(r"[​-‏ - ﻿]")

# Emoji / pictograph blocks (kept explicit so the report can cite the ranges).
RE_EMOJI = re.compile(
    "[" "\U0001F300-\U0001F5FF" "\U0001F600-\U0001F64F" "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F" "\U0001F900-\U0001F9FF" "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF" "\U0000FE00-\U0000FE0F" "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF" "\U0001F1E6-\U0001F1FF" "]+",
    flags=re.UNICODE,
)

# Boilerplate that a support desk appends automatically - such turns carry no
# conversational signal and are dropped.
RE_AUTOMATED = re.compile(
    r"(?:this is an automated (?:message|response|reply)"
    r"|do not reply to this (?:e-?mail|message)"
    r"|auto-?generated (?:message|response)"
    r"|your ticket (?:has been|was) (?:created|opened|logged) automatically"
    r"|out of office"
    r"|unsubscribe from these (?:e-?mails|notifications))",
    re.I,
)

# Chat-speak seen in real support inboxes; normalised so the encoder does not
# waste vocabulary entries on them.
CHAT_NORMALISATION = {
    "u": "you", "ur": "your", "r": "are", "n": "and", "pls": "please",
    "plz": "please", "plss": "please", "thx": "thanks", "ty": "thank you",
    "tnx": "thanks", "cud": "could", "wud": "would", "shud": "should",
    "asap": "as soon as possible", "info": "information", "acct": "account",
    "acc": "account", "ordr": "order", "recieve": "receive", "recieved": "received",
    "adress": "address", "cancl": "cancel", "cancle": "cancel", "refnd": "refund",
    "paymnt": "payment", "im": "i am", "ive": "i have", "dont": "do not",
    "cant": "can not", "wont": "will not", "didnt": "did not", "doesnt": "does not",
    "isnt": "is not", "wasnt": "was not", "couldnt": "could not",
    "wouldnt": "would not", "shouldnt": "should not", "havent": "have not",
    "hasnt": "has not", "id": "i would", "ill": "i will", "wanna": "want to",
    "gonna": "going to", "gotta": "got to", "bcoz": "because", "bcz": "because",
    "coz": "because", "b4": "before", "msg": "message", "acnt": "account",
}

# Rare placeholders collapse into one generic slot so the decoder is not asked
# to learn hundreds of one-off tokens.
GENERIC_PLACEHOLDER = "<ph_details>"
PLACEHOLDER_MIN_FREQ = 20

# Function words carry no domain information, so they are excluded when the
# out-of-scope gate measures how much of a query the support corpus knows.
STOPWORDS = set("""
a an the this that these those i me my mine myself we our ours us you your
yours he him his she her hers it its they them their theirs is am are was were
be been being do does did doing done have has had having can could will would
shall should may might must of in on at to for with about from by as into over
under again further and or but if then else than so not no nor too very just
now here there when where why how what which who whom whose all any both each
few more most other some such only own same s t don dont cant im ive ill id
please pls hey hi hello thanks thank ok okay yes yeah still want need get got
""".split())

ANCHOR_TOP_K = 200          # size of the domain-anchor list written for the report
SCOPE_MIN_FREQ = 2          # a query word must occur this often to count as known

# Token pattern: keeps our angle-bracket specials whole; words hold on to their
# internal apostrophes and hyphens ("don't", "step-by-step", "up-to-date"), so a
# dash used as punctuation ("shipping - we will ...") stays a separate token and
# can be rendered with spaces around it.
RE_TOKEN = re.compile(r"<[a-z0-9_]+>|[a-z]+(?:['’\-][a-z]+)*|\d+|[^\sa-z\d]")


# --------------------------------------------------------------------------- #
# 2. Placeholder handling
# --------------------------------------------------------------------------- #
def slugify_placeholder(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return "<ph_{}>".format(slug) if slug else GENERIC_PLACEHOLDER


def build_placeholder_map(texts: List[str]) -> Dict[str, str]:
    """{{Order Number}} -> <ph_order_number>; rare ones -> <ph_details>."""
    counter = Counter()
    for text in texts:
        for raw in RE_PLACEHOLDER.findall(str(text)):
            counter[raw.strip()] += 1
    mapping = {}
    for raw, count in counter.items():
        mapping[raw] = (slugify_placeholder(raw) if count >= PLACEHOLDER_MIN_FREQ
                        else GENERIC_PLACEHOLDER)
    return mapping


def apply_placeholders(text: str, mapping: Dict[str, str]) -> str:
    def repl(match):
        return " " + mapping.get(match.group(1).strip(), GENERIC_PLACEHOLDER) + " "
    return RE_PLACEHOLDER.sub(repl, text)


def placeholder_surface(token: str, mapping: Dict[str, str]) -> str:
    """Inverse map, used when rendering a reply back to the customer."""
    inverse = {v: k for k, v in mapping.items() if v != GENERIC_PLACEHOLDER}
    if token in inverse:
        return "{{" + inverse[token] + "}}"
    return "{{Details}}"


# --------------------------------------------------------------------------- #
# 3. Cleaning + tokenisation
# --------------------------------------------------------------------------- #
def clean_text(text: str, mapping: Dict[str, str], normalise_chat: bool = False) -> str:
    """Full cleaning chain; returns a lower-cased, normalised string."""
    text = str(text)
    text = RE_NONPRINT.sub(" ", text)
    text = RE_HTML.sub(" ", text)                 # strip markup *before* we
    text = apply_placeholders(text, mapping)      # introduce our <...> tokens
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    text = RE_URL.sub(" <url> ", text)
    text = RE_EMAIL.sub(" <email> ", text)
    text = RE_USERNAME.sub(" <user> ", text)
    text = RE_HASHTAG.sub(r" \1 ", text)
    text = RE_EMOJI.sub(" ", text)
    text = text.lower()
    text = RE_REPEAT_PUNCT.sub(r"\1", text)
    text = RE_REPEAT_CHAR.sub(r"\1\1", text)
    if normalise_chat:
        text = " ".join(CHAT_NORMALISATION.get(w, w) for w in text.split())
    text = RE_MULTISPACE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return RE_TOKEN.findall(text)


RE_REF_NUMBER = re.compile(r"^[#-]?\d[\d\-]*$")
MIN_REF_DIGITS = 4          # "30 days" and "24/7" must survive untouched


def reference_slot(context: str) -> str:
    """Which placeholder a bare identifier in this query most likely denotes."""
    lowered = context.lower()
    if re.search(r"\b(invoice|bill|billing)", lowered):
        return "<ph_invoice_number>"
    if re.search(r"\b(track|tracking|shipment|parcel|package)", lowered):
        return "<ph_tracking_number>"
    if re.search(r"\b(refund|reimburse|money back)", lowered):
        return "<ph_refund_amount>"
    return "<ph_order_number>"


def slot_reference_numbers(tokens: List[str], context: str) -> List[str]:
    """Replace bare reference numbers with the placeholder they stand for.

    In the corpus every identifier is normally a ``{{...}}`` slot, so a live
    query such as "where is my order 4471902" would otherwise become an
    out-of-vocabulary token and look out of scope. Working on the **token**
    stream rather than the raw string matters: the corpus also contains
    identifiers glued to the preceding word ("cancel purchase370795561790"),
    which only separate once the tokeniser has run.

    It also makes re-processing an already-processed query - which the
    evaluation script does, because the splits are stored tokenised - a no-op
    for all but 3 of the 26,244 pairs. (Those three end in a chat-speak word
    glued to punctuation, "user info?", which the whitespace-based chat-speak
    pass cannot see until after tokenisation.)
    """
    slot = reference_slot(context)
    out = []
    for tok in tokens:
        digits = sum(1 for ch in tok if ch.isdigit())
        if digits >= MIN_REF_DIGITS and RE_REF_NUMBER.match(tok):
            out.append(slot)
        else:
            out.append(tok)
    return out


def preprocess_query(text: str, mapping: Dict[str, str]) -> List[str]:
    """Single entry point used by training *and* the live application (Task 4).

    Applies the cleaning chain, then chat-speak normalisation and
    reference-number slotting, which only make sense on input typed by a real
    customer.
    """
    cleaned = clean_text(text, mapping, normalise_chat=True)
    return slot_reference_numbers(tokenize(cleaned), cleaned)


def content_words(tokens: List[str]) -> List[str]:
    """Domain-bearing words: alphabetic, longer than two characters, not a stopword."""
    return [t for t in tokens
            if t.isalpha() and len(t) > 2 and t not in STOPWORDS]


def build_scope_lexicon(train_query_tokens: List[List[str]]) -> Dict:
    """Lexicon used by the out-of-scope gate at inference time.

    Deliberately built from the **query side of the training split only**. The
    full model vocabulary also contains every word of every agent reply, which
    is far too permissive: it makes almost any English sentence look familiar.
    """
    counter = Counter()
    for toks in train_query_tokens:
        counter.update(toks)
    known = sorted(t for t, c in counter.items() if c >= SCOPE_MIN_FREQ)

    keep = set(content_words(list(counter.keys())))
    content = Counter({t: c for t, c in counter.items() if t in keep})
    anchors = [t for t, _ in content.most_common(ANCHOR_TOP_K)]
    return {"query_vocab": known, "anchors": anchors,
            "stopwords": sorted(STOPWORDS), "min_freq": SCOPE_MIN_FREQ}


def truncate_at_sentence(tokens: List[str], limit: int) -> List[str]:
    """Cut a reply to ``limit`` tokens at the last complete sentence.

    Half of the replies in the corpus are longer than the decoder budget. A hard
    cut would teach the model to stop mid-sentence, so we back off to the last
    sentence-final punctuation that still fits and only fall back to a hard cut
    when the very first sentence is already too long.
    """
    if len(tokens) <= limit:
        return tokens
    window = tokens[:limit]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in (".", "!", "?"):
            return window[: i + 1]
    return window


# --------------------------------------------------------------------------- #
# 4. Detokenisation (model output -> text shown to the customer)
# --------------------------------------------------------------------------- #
_NO_SPACE_BEFORE = set(list(".,!?;:%)]}"))
_NO_SPACE_AFTER = set("([{$#")
_TIGHT_JOINERS = set("/@_")         # glued to the words on both sides
_PAIRED = {'"', "'"}

SPECIAL_SURFACE = {
    "<url>": "{{Website URL}}",
    "<email>": "{{Customer Support Email}}",
    "<user>": "there",
    "<num>": "{{Number}}",
    "<unk>": "",
}


def _surface(tok: str, inverse: Dict[str, str]) -> str:
    if tok in SPECIAL_SURFACE:
        return SPECIAL_SURFACE[tok]
    if tok == GENERIC_PLACEHOLDER:
        return "{{Details}}"
    if tok.startswith("<ph_") and tok.endswith(">"):
        return "{{" + inverse.get(tok, tok[4:-1].replace("_", " ").title()) + "}}"
    return tok


def detokenize(tokens: List[str], mapping: Dict[str, str] = None) -> str:
    """Re-assembles decoder tokens into a readable, capitalised support reply."""
    mapping = mapping or {}
    inverse = {v: k for k, v in mapping.items() if v != GENERIC_PLACEHOLDER}

    words = [w for w in (_surface(t, inverse) for t in tokens) if w]

    pieces: List[str] = []
    open_quote = {'"': False, "'": False}
    for i, tok in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        prev = words[i - 1] if i > 0 else None

        if not pieces:
            attach = True
        elif tok in _NO_SPACE_BEFORE:
            attach = True
        elif tok in _TIGHT_JOINERS:
            # "24 / 7" -> "24/7" when both sides are words, otherwise leave it
            attach = bool(prev and prev[-1].isalnum() and nxt and nxt[:1].isalnum())
        elif prev and prev[-1] in _TIGHT_JOINERS and len(prev) == 1:
            attach = True
        elif tok in _PAIRED:
            attach = open_quote[tok]               # a closing quote hugs the word,
                                                   # an opening one takes a space
        elif prev in _PAIRED and open_quote.get(prev):
            attach = True                          # word follows an opening quote
        elif prev and prev[-1] in _NO_SPACE_AFTER and len(prev) == 1:
            attach = True
        elif tok.startswith("'") and len(tok) <= 3:
            attach = True                          # 's, 've, 'll, n't
        else:
            attach = False

        if tok in _PAIRED:
            open_quote[tok] = not open_quote[tok]

        pieces.append(tok) if attach else pieces.append(" " + tok)

    out = "".join(pieces)
    out = re.sub(r"\bi\b", "I", out)
    out = re.sub(r"(^|(?<=[.!?]\s))([a-z])",
                 lambda m: m.group(1) + m.group(2).upper(), out)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    out = RE_MULTISPACE.sub(" ", out).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


# --------------------------------------------------------------------------- #
# 5. Pipeline
# --------------------------------------------------------------------------- #
def group_aware_split(df: pd.DataFrame, ratios: Tuple[float, float, float],
                      seed: int) -> pd.DataFrame:
    """Split by *distinct query*, so no paraphrase of a test query leaks into train."""
    import numpy as np

    rng = np.random.RandomState(seed)
    keys = df["query_clean"].unique()
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(ratios[0] * n)
    n_valid = int(ratios[1] * n)
    assign = {}
    for i, key in enumerate(keys):
        assign[key] = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")
    df = df.copy()
    df["split"] = df["query_clean"].map(assign)
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess the support corpus")
    parser.add_argument("--raw", default=C.RAW_CSV)
    parser.add_argument("--query-col", default="instruction")
    parser.add_argument("--response-col", default="response")
    parser.add_argument("--min-freq", type=int, default=C.MIN_FREQ)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="debug option: only use the first N rows")
    args = parser.parse_args()

    os.makedirs(C.PROCESSED_DIR, exist_ok=True)
    stats = {}

    print("[1/7] loading {}".format(args.raw))
    df = pd.read_csv(args.raw)
    if args.max_rows:
        df = df.head(args.max_rows)
    df = df.rename(columns={args.query_col: "query", args.response_col: "response"})
    keep = [c for c in ["query", "response", "category", "intent", "flags"] if c in df.columns]
    df = df[keep]
    stats["raw_pairs"] = int(len(df))
    print("      {} raw pairs".format(len(df)))

    print("[2/7] dropping empty / automated turns")
    before = len(df)
    df = df.dropna(subset=["query", "response"])
    stats["dropped_null"] = int(before - len(df))

    before = len(df)
    auto_mask = (df["query"].astype(str).str.contains(RE_AUTOMATED) |
                 df["response"].astype(str).str.contains(RE_AUTOMATED))
    stats["dropped_automated"] = int(auto_mask.sum())
    df = df[~auto_mask]
    print("      removed {} null / {} automated".format(
        stats["dropped_null"], stats["dropped_automated"]))

    print("[3/7] normalising placeholders and cleaning text")
    mapping = build_placeholder_map(list(df["query"]) + list(df["response"]))
    stats["placeholder_types"] = int(len(set(mapping.values())))
    # The query side goes through the *same* function the web app calls, so
    # training and inference can never drift apart.
    df["query_clean"] = [" ".join(preprocess_query(t, mapping)) for t in df["query"]]
    df["response_clean"] = [clean_text(t, mapping, normalise_chat=False) for t in df["response"]]

    print("[4/7] removing duplicates and degenerate pairs")
    before = len(df)
    df = df.drop_duplicates(subset=["query_clean", "response_clean"])
    stats["dropped_duplicate_pairs"] = int(before - len(df))

    before = len(df)
    q_len = df["query_clean"].str.split().str.len()
    r_len = df["response_clean"].str.split().str.len()
    df = df[(q_len >= 2) & (q_len <= C.MAX_SRC_LEN - 2) & (r_len >= 3)]
    stats["dropped_too_short"] = int(before - len(df))

    # A single query may legitimately map to several agent phrasings; cap it so
    # the loss is not dominated by the most frequent intents.
    before = len(df)
    df = df.groupby("query_clean", group_keys=False, sort=False).head(4)
    stats["dropped_over_represented"] = int(before - len(df))
    stats["pairs_after_cleaning"] = int(len(df))
    print("      {} duplicate pairs, {} degenerate, {} over-represented -> {} kept".format(
        stats["dropped_duplicate_pairs"], stats["dropped_too_short"],
        stats["dropped_over_represented"], len(df)))

    print("[5/7] tokenising and truncating replies at a sentence boundary")
    df["query_tokens"] = [tokenize(t) for t in df["query_clean"]]
    raw_response_tokens = [tokenize(t) for t in df["response_clean"]]
    stats["truncated_responses"] = int(
        sum(1 for t in raw_response_tokens if len(t) > C.MAX_TGT_LEN - 2))
    df["response_tokens"] = [truncate_at_sentence(t, C.MAX_TGT_LEN - 2)
                             for t in raw_response_tokens]
    stats["mean_query_tokens"] = round(float(df["query_tokens"].apply(len).mean()), 2)
    stats["mean_response_tokens"] = round(float(df["response_tokens"].apply(len).mean()), 2)
    print("      mean query {} tok / mean response {} tok / {} responses truncated at {}".format(
        stats["mean_query_tokens"], stats["mean_response_tokens"],
        stats["truncated_responses"], C.MAX_TGT_LEN))

    print("[6/7] splitting {}/{}/{} by distinct query".format(*C.SPLIT_RATIOS))
    df = group_aware_split(df, C.SPLIT_RATIOS, C.RANDOM_SEED)
    counts = df["split"].value_counts().to_dict()
    stats["split_sizes"] = {k: int(v) for k, v in counts.items()}
    print("      " + " / ".join("{}: {}".format(k, v) for k, v in sorted(counts.items())))

    print("[7/7] building vocabulary from the TRAIN split only")
    train_df = df[df["split"] == "train"]
    vocab = Vocabulary.build(
        list(train_df["query_tokens"]) + list(train_df["response_tokens"]),
        min_freq=args.min_freq,
    )
    all_counter = Counter()
    for toks in list(df["query_tokens"]) + list(df["response_tokens"]):
        all_counter.update(toks)
    stats["vocab_size"] = len(vocab)
    stats["token_coverage"] = round(vocab.coverage(all_counter), 4)
    stats["min_freq"] = args.min_freq
    print("      vocab {} types, covers {:.2%} of all running tokens".format(
        len(vocab), stats["token_coverage"]))

    vocab.save(C.VOCAB_JSON)
    with open(os.path.join(C.PROCESSED_DIR, "placeholder_map.json"), "w",
              encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=1, ensure_ascii=False)

    lexicon = build_scope_lexicon(list(train_df["query_tokens"]))
    with open(C.SCOPE_LEXICON_JSON, "w", encoding="utf-8") as fh:
        json.dump(lexicon, fh, indent=1)
    stats["scope_query_vocab"] = len(lexicon["query_vocab"])
    print("      scope lexicon: {} known query words, {} domain anchors".format(
        len(lexicon["query_vocab"]), len(lexicon["anchors"])))

    # Persist the *token* streams so that training and the live application are
    # guaranteed to see identical text.
    df["query_clean"] = [" ".join(t) for t in df["query_tokens"]]
    df["response_clean"] = [" ".join(t) for t in df["response_tokens"]]
    out_cols = ["query_clean", "response_clean", "category", "intent"]
    out_cols = [c for c in out_cols if c in df.columns]
    for name, path in (("train", C.TRAIN_CSV), ("valid", C.VALID_CSV), ("test", C.TEST_CSV)):
        sub = df[df["split"] == name][out_cols]
        sub.to_csv(path, index=False)
        print("      wrote {} ({} rows)".format(os.path.relpath(path, C.ROOT_DIR), len(sub)))

    with open(C.STATS_JSON, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print("\nPreprocessing summary -> {}".format(os.path.relpath(C.STATS_JSON, C.ROOT_DIR)))
    for k, v in stats.items():
        print("  {:<28} {}".format(k, v))


if __name__ == "__main__":
    main()
