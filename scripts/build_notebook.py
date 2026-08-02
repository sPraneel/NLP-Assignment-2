"""Regenerate ``customer_support_response_generation.ipynb``.

This is a convenience tool, not a dependency: the notebook it writes is
completely self-contained - it defines every function it uses inline and reads
or writes nothing except the ``data/`` folder. Nothing in ``scripts/`` has to be
importable for the notebook to run.

    python scripts/build_notebook.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "customer_support_response_generation.ipynb")

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# =========================================================================== #
# 0. Title and setup
# =========================================================================== #
md(r'''
# Customer Support Response Generation Chatbot

**M.Tech. AIML — Natural Language Processing (S2-25_AIMLCZG530)**
Assignment 2 · GS-3 · Encoder–Decoder Conversational Response Generation

### Group details

| Name | BITS ID | Contribution |
|---|---|---|
| _&lt;member 1&gt;_ | _&lt;2xxxxxxx&gt;_ | 25% |
| _&lt;member 2&gt;_ | _&lt;2xxxxxxx&gt;_ | 25% |
| _&lt;member 3&gt;_ | _&lt;2xxxxxxx&gt;_ | 25% |
| _&lt;member 4&gt;_ | _&lt;2xxxxxxx&gt;_ | 25% |

> Replace the placeholders above with the actual names, BITS IDs and
> contribution percentages before submitting.

---

### What this notebook contains

| Section | Assignment task |
|---|---|
| 1 | Problem analysis — domain, requirements, I/O, scope handling |
| 2 | Data collection and preprocessing |
| 3 | Model development — encoder–decoder, training, decoding |
| 4 | Application development — Streamlit web app (with screenshots) |
| 5 | Evaluation and demonstration — BLEU / ROUGE / perplexity + manual rating |
| 6 | Observations, conclusion and references |

### How to run it

This notebook is **self-contained**: every class and function it uses is defined
in the cells below, and the only thing it touches on disk is the `data/` folder
beside it. There is nothing to install from this project and nothing to import —
*Run All* is enough.

```
project/
├── customer_support_response_generation.ipynb   <- this notebook
├── app.py                                       <- Streamlit demo (Section 4)
├── data/                                        <- corpus + every artefact
└── scripts/                                     <- optional command-line copies
```

`app.py` and `scripts/` exist for convenience (training from a terminal, serving
the web app). **The notebook never imports them.**

Three switches in the setup cell decide how much is recomputed: the
preprocessing is rebuilt from the raw corpus every time, while the trained model
and the metrics are reused from `data/` when they are already there.
''')

code(r'''
# --------------------------------------------------------------------------- #
# Setup: imports, paths and the switches that decide what gets recomputed.
# --------------------------------------------------------------------------- #
# On a fresh machine (a new Colab runtime, say) install the dependencies once:
#     %pip install -q torch pandas numpy matplotlib sacrebleu rouge-score

import glob
import json
import math
import os
import random
import re
import sys
import textwrap
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

# ---- paths: everything the notebook reads or writes lives in data/ --------- #
ROOT = os.getcwd()
if not os.path.isdir(os.path.join(ROOT, "data")):
    parent = os.path.dirname(ROOT)              # notebook opened from a subfolder
    if os.path.isdir(os.path.join(parent, "data")):
        ROOT = parent
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_CSV            = os.path.join(DATA_DIR, "bitext_customer_support_27k.csv")
TRAIN_CSV          = os.path.join(DATA_DIR, "train.csv")
VALID_CSV          = os.path.join(DATA_DIR, "valid.csv")
TEST_CSV           = os.path.join(DATA_DIR, "test.csv")
VOCAB_JSON         = os.path.join(DATA_DIR, "vocab.json")
PLACEHOLDER_JSON   = os.path.join(DATA_DIR, "placeholder_map.json")
SCOPE_LEXICON_JSON = os.path.join(DATA_DIR, "scope_lexicon.json")
STATS_JSON         = os.path.join(DATA_DIR, "preprocessing_stats.json")
CHECKPOINT         = os.path.join(DATA_DIR, "best_model.pt")
HISTORY_JSON       = os.path.join(DATA_DIR, "history.json")
METRICS_JSON       = os.path.join(DATA_DIR, "metrics.json")
LOSS_CURVE_PNG     = os.path.join(DATA_DIR, "loss_curve.png")
RATING_SHEET_CSV   = os.path.join(DATA_DIR, "manual_rating_sheet.csv")
SAMPLE_QUERIES_TXT = os.path.join(DATA_DIR, "sample_queries.txt")
SCREENSHOT_DIR     = os.path.join(DATA_DIR, "screenshots")

# ---- what to recompute ---------------------------------------------------- #
REBUILD_DATA      = True    # preprocessing: ~1 minute, fully deterministic
RETRAIN_MODEL     = False   # training: ~30 min on MPS/GPU, else reuse best_model.pt
RECOMPUTE_METRICS = False   # evaluation: ~5 min, else reuse metrics.json
EVAL_LIMIT        = 400     # test queries decoded during evaluation

print("project root :", ROOT)
print("data folder  :", DATA_DIR)
print("python       :", sys.version.split()[0])
print("torch        :", torch.__version__)
print("raw corpus   :", "found" if os.path.exists(RAW_CSV) else "MISSING")
''')

code(r'''
# --------------------------------------------------------------------------- #
# Hyper-parameters and constants - the single source of truth for the notebook.
# --------------------------------------------------------------------------- #
# Special tokens, pinned to ids 0-3 so any checkpoint stays readable.
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

# Data preparation
MAX_SRC_LEN  = 32               # customer query, in tokens (incl. <sos>/<eos>)
MAX_TGT_LEN  = 120              # agent reply, in tokens (incl. <sos>/<eos>)
MIN_FREQ     = 2                # a word must appear this often to enter the vocabulary
SPLIT_RATIOS = (0.80, 0.10, 0.10)
RANDOM_SEED  = 42

# Model
ARCH = "transformer"            # "transformer" | "lstm_attn"
D_MODEL, N_HEADS = 256, 4
N_ENC_LAYERS, N_DEC_LAYERS, FFN_DIM = 3, 3, 512
DROPOUT = 0.1
EMB_DIM, HIDDEN_DIM, LSTM_LAYERS = 256, 512, 1      # LSTM + Bahdanau variant

# Training
BATCH_SIZE, EPOCHS = 64, 25
LEARNING_RATE, WEIGHT_DECAY = 5e-4, 1e-4
LABEL_SMOOTHING, CLIP_NORM = 0.1, 1.0
PATIENCE, WARMUP_STEPS = 4, 400

# Decoding
DECODE_STRATEGY = "beam"
BEAM_SIZE, LENGTH_PENALTY = 3, 0.7
MAX_DECODE_LEN, NO_REPEAT_NGRAM = 120, 3

# Out-of-scope gate (Section 1.2, FR-6)
OOS_OOV_RATIO       = 0.50      # >50% unseen content words -> out of scope
OOS_MIN_AVG_LOGPROB = -1.50     # decoder unsure            -> out of scope
OOS_MESSAGE = (
    "I'm sorry, I'm not able to help with that one. I'm a customer-support "
    "assistant and I can answer questions about orders, refunds, payments, "
    "invoices, shipping and delivery, subscriptions, account access and "
    "contacting a human agent. Could you rephrase your question around one of "
    "those topics, or type 'agent' and I'll hand you over to a colleague?"
)
IN_SCOPE_CATEGORIES = [
    "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE",
    "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
]

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print("configured: {} architecture, src {} tok / tgt {} tok, seed {}".format(
    ARCH, MAX_SRC_LEN, MAX_TGT_LEN, RANDOM_SEED))
''')

# =========================================================================== #
# 1. Problem analysis
# =========================================================================== #
md(r'''
---
# 1. Problem Analysis

## 1.1 Application domain and business processes

The domain is **customer support / service operations** for a consumer-facing
retail or subscription business. Support desks receive a high volume of
repetitive contacts, and an agent spends most of a shift re-typing near-identical
replies.

The application is an **agent-assist response drafter**. It supports these
business processes:

| Business process | Typical query | Where the draft helps |
|---|---|---|
| Order management | *"I want to cancel order 4471902"* | first-response drafting, order-change acknowledgement |
| Returns & refunds | *"How do I get a refund?"* | refund-policy explanation, status updates |
| Billing & invoicing | *"There is a wrong charge on my invoice"* | invoice retrieval instructions, dispute intake |
| Payments | *"What payment methods do you accept?"* | payment-method FAQ, failed-payment guidance |
| Shipping & delivery | *"Where is my package?"* | tracking guidance, delivery-window explanation |
| Account access | *"I forgot my password"* | recovery walk-through, security guidance |
| Subscription management | *"Upgrade my account to premium"* | plan-change instructions |
| Contact / escalation | *"I need a human agent"* | routing and hand-off |
| Feedback & complaints | *"I want to leave feedback"* | acknowledgement and routing |

Business value: lower average handling time, consistent tone, faster first
response, and lower cost per ticket. The agent stays in the loop — the model
drafts, a human approves.

## 1.2 Problem statement and functional requirements

> **Problem statement.** Given a free-text customer support query, automatically
> generate a fluent, contextually relevant support reply using an encoder–decoder
> neural network, rather than selecting a fixed template from a bank.

**Functional requirements**

| # | Requirement |
|---|---|
| FR-1 | Accept a free-text query (5–100 words) typed into a chat box. |
| FR-2 | Accept a `.txt` (one query per line) or `.csv` (a query column) upload and answer every row. |
| FR-3 | Generate the reply token by token with a trained encoder–decoder; **no** retrieval or template lookup. |
| FR-4 | Support greedy **and** beam-search decoding, selectable at run time. |
| FR-5 | Display the reply in a chat interface and keep the full transcript on screen. |
| FR-6 | Detect out-of-scope queries and return a hand-off message instead of an invented answer. |
| FR-7 | Expose per-reply diagnostics (confidence, unknown-word ratio, scope decision, latency). |
| FR-8 | Allow the transcript and the batch results to be downloaded. |

**Non-functional requirements:** a reply in under ~3 s on CPU; the app runs
locally with `streamlit run app.py`; the model is < 10 M parameters so it
trains on a laptop; all preprocessing is deterministic under a fixed seed.

**Out-of-scope behaviour (FR-6).** A query is refused when *either*
* more than **50 %** of its *content* words (non-stopword, longer than two
  characters) never occur in the **queries** of the training split, or
* the decoder's mean per-token log-probability is below **−1.5** (the model is
  guessing).

The user then receives a message that names the topics the assistant *can* handle
and offers a hand-off to a human agent. This is deliberately conservative: in
customer communication, a wrong confident answer costs more than an admission of
ignorance.

## 1.3 Expected input and output

| | |
|---|---|
| **Input** | One customer query as free text (chat box), or a `.txt`/`.csv` file of queries (batch mode). |
| **Output** | A generated support reply of roughly 40–120 words, rendered with readable `{{Placeholders}}` (e.g. `{{Order Number}}`) that a human agent fills in, plus a scope flag and confidence score. Batch mode returns a downloadable CSV. |
| **Conversation type** | **Single-turn.** Each query is encoded and answered independently. The transcript is retained and displayed on screen (and downloadable), but earlier turns are not fed back into the encoder. |

Multi-turn support would require the corpus to be re-flattened with dialogue
history concatenated into the source sequence; the architecture supports it, the
chosen corpus does not contain the history.
''')

# =========================================================================== #
# 2. Data
# =========================================================================== #
md(r'''
---
# 2. Data Collection and Preprocessing

## 2.1 Dataset

| | |
|---|---|
| **Name** | Bitext – Customer Service Tagged Training Dataset for LLM-based Virtual Assistants |
| **Source** | <https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset> |
| **Licence** | Community Data License Agreement – Sharing, v1.0 (**CDLA-Sharing-1.0**) |
| **Size** | 26,872 query–response pairs, 18 MB, one CSV file |
| **Structure** | `instruction` = customer message, `response` = agent reply, plus `category`, `intent`, `flags` metadata |

It satisfies the dataset description in the problem statement: query–response
pairs, 10 k–1 L rows, support-desk messages, with optional intent metadata. The
file is stored as `data/bitext_customer_support_27k.csv`.
''')

code(r'''
raw = pd.read_csv(RAW_CSV)
print("shape:", raw.shape)
print("columns:", list(raw.columns))
raw.head(3)
''')

code(r'''
summary = pd.DataFrame({
    "customer query (words)": raw["instruction"].str.split().str.len().describe(),
    "agent reply (words)":    raw["response"].str.split().str.len().describe(),
}).round(2)
print(summary.to_string())
print("\ndistinct intents   :", raw["intent"].nunique())
print("distinct categories:", raw["category"].nunique())
print()
print(raw["category"].value_counts().to_string())
''')

md(r'''
The `flags` column encodes the linguistic variation deliberately injected into
each query — `Q` colloquial, `Z` noise/typos, `L` offensive-free paraphrase,
`M` morphological variation, and so on. That is useful: the encoder sees
misspelled and informal phrasings of the same intent, which is what a real
support inbox looks like.
''')

code(r'''
print(raw["flags"].value_counts().head(8).to_string())
for i in [0, 2, 7]:
    print("\n--- row", i, "| intent:", raw["intent"][i], "| flags:", raw["flags"][i])
    print("Q:", raw["instruction"][i])
    print("A:", textwrap.fill(raw["response"][i][:400], 100))
''')

md(r'''
## 2.2 Preprocessing

The next three cells define the complete preprocessing chain — regular
expressions, cleaning, tokenisation, detokenisation and the vocabulary — and the
fourth runs it end to end over the corpus. The cells after that demonstrate each
step on real rows.

**a) Placeholder normalisation.** The corpus marks variable data as
`{{Order Number}}`, `{{Account Type}}`, … Left alone, the braces would be
shredded by tokenisation. Each of the 43 frequent slots becomes **one**
vocabulary token; the long tail collapses into `<ph_details>`. The mapping is
saved so replies can be rendered back with readable placeholders.

**b) Cleaning and normalisation.** HTML tags, URLs, e-mail addresses,
`@usernames`, hashtags, emoji and zero-width characters are removed or replaced
by typed tokens; curly quotes/dashes are normalised; repeated characters
(`heyyyy`) and punctuation (`???`) are collapsed; everything is lower-cased.
Chat-speak is expanded on the query side only.

**c) Removing automated and duplicate messages.** A regex filter drops
auto-replies ("this is an automated message", "do not reply", out-of-office
notices); exact duplicate `(query, reply)` pairs are removed; degenerate rows
(fewer than 2 query tokens or 3 reply tokens) are dropped; and at most **4**
replies are kept per distinct query so that the most frequent intents do not
dominate the loss.

**d) Sentence-aware truncation.** Half the replies are longer than the 120-token
decoder budget. A hard cut would train the model to stop mid-sentence, so the
reply is cut at the **last complete sentence** that fits.

**e) Special tokens, padding and the splits.** `<pad> <sos> <eos> <unk>` are
pinned to ids 0–3. Queries are padded/truncated to 32 tokens and replies to 120.
The split is **group-aware**: every copy of a given query lands in the same
split, so no paraphrase leaks from train into test. The vocabulary is built from
the **training split only**, with a minimum frequency of 2.

**f) A separate scope lexicon.** The out-of-scope gate must not use the model
vocabulary — that vocabulary contains every word of every *agent reply*, so
almost any English sentence looks familiar to it. A second, much tighter lexicon
is therefore built from the training **queries** only.
''')

code(r'''
# =========================================================================== #
# Preprocessing chain - regular expressions and lookup tables
# =========================================================================== #
RE_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_USERNAME = re.compile(r"(?<![\w])@\w{2,}")
RE_HASHTAG = re.compile(r"(?<![\w])#(\w+)")
# Strips markup, but never the typed tokens this pipeline itself introduces -
# without the lookahead, running the cleaner over already-processed text (as the
# evaluation does, since the splits are stored tokenised) would silently delete
# every <ph_...> slot.
RE_HTML = re.compile(
    r"<(?!/?(?:ph_[a-z0-9_]*|url|email|user|num|unk|pad|sos|eos)>)[^<>]{1,40}>")
RE_MULTISPACE = re.compile(r"\s+")
RE_REPEAT_CHAR = re.compile(r"(.)\1{2,}")           # "heyyyy" -> "heyy"
RE_REPEAT_PUNCT = re.compile(r"([!?.,])\1{1,}")     # "???"    -> "?"
RE_NONPRINT = re.compile("[​-‏‪-‮﻿]")

# Emoji / pictograph blocks (kept explicit so the report can cite the ranges).
RE_EMOJI = re.compile(
    "[" "\U0001F300-\U0001F5FF" "\U0001F600-\U0001F64F" "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F" "\U0001F900-\U0001F9FF" "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF" "\U0000FE00-\U0000FE0F" "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF" "\U0001F1E6-\U0001F1FF" "]+",
    flags=re.UNICODE,
)

# Boilerplate a support desk appends automatically - no conversational signal.
RE_AUTOMATED = re.compile(
    r"(?:this is an automated (?:message|response|reply)"
    r"|do not reply to this (?:e-?mail|message)"
    r"|auto-?generated (?:message|response)"
    r"|your ticket (?:has been|was) (?:created|opened|logged) automatically"
    r"|out of office"
    r"|unsubscribe from these (?:e-?mails|notifications))",
    re.I,
)

# Token pattern: keeps our angle-bracket specials whole; words hold on to their
# internal apostrophes and hyphens ("don't", "step-by-step").
RE_TOKEN = re.compile("<[a-z0-9_]+>|[a-z]+(?:['’\\-][a-z]+)*|\\d+|[^\\sa-z\\d]")

# Chat-speak seen in real support inboxes; normalised so the encoder does not
# waste vocabulary entries on it.
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

GENERIC_PLACEHOLDER = "<ph_details>"
PLACEHOLDER_MIN_FREQ = 20      # rarer slots collapse into <ph_details>
ANCHOR_TOP_K = 200             # size of the domain-anchor list
SCOPE_MIN_FREQ = 2             # a query word must occur this often to count as known

print("{} chat-speak entries, {} stopwords, {} regexes".format(
    len(CHAT_NORMALISATION), len(STOPWORDS), 9))
''')

code(r'''
# =========================================================================== #
# Preprocessing chain - cleaning, tokenisation, detokenisation  (2.2 a-d, f)
# =========================================================================== #
def slugify_placeholder(name):
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return "<ph_{}>".format(slug) if slug else GENERIC_PLACEHOLDER


def build_placeholder_map(texts):
    """{{Order Number}} -> <ph_order_number>; rare ones -> <ph_details>."""
    counter = Counter()
    for text in texts:
        for slot in RE_PLACEHOLDER.findall(str(text)):
            counter[slot.strip()] += 1
    return {slot: (slugify_placeholder(slot) if count >= PLACEHOLDER_MIN_FREQ
                   else GENERIC_PLACEHOLDER)
            for slot, count in counter.items()}


def apply_placeholders(text, mapping):
    def repl(match):
        return " " + mapping.get(match.group(1).strip(), GENERIC_PLACEHOLDER) + " "
    return RE_PLACEHOLDER.sub(repl, text)


def clean_text(text, mapping, normalise_chat=False):
    """Full cleaning chain; returns a lower-cased, normalised string."""
    text = str(text)
    text = RE_NONPRINT.sub(" ", text)
    text = RE_HTML.sub(" ", text)                 # strip markup *before* we
    text = apply_placeholders(text, mapping)      # introduce our <...> tokens
    text = text.replace("’", "'").replace("‘", "'")
    text = (text.replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-"))
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
    return RE_MULTISPACE.sub(" ", text).strip()


def tokenize(text):
    return RE_TOKEN.findall(text)


RE_REF_NUMBER = re.compile(r"^[#-]?\d[\d\-]*$")
MIN_REF_DIGITS = 4             # "30 days" and "24/7" must survive untouched


def reference_slot(context):
    """Which placeholder a bare identifier in this query most likely denotes."""
    lowered = context.lower()
    if re.search(r"\b(invoice|bill|billing)", lowered):
        return "<ph_invoice_number>"
    if re.search(r"\b(track|tracking|shipment|parcel|package)", lowered):
        return "<ph_tracking_number>"
    if re.search(r"\b(refund|reimburse|money back)", lowered):
        return "<ph_refund_amount>"
    return "<ph_order_number>"


def slot_reference_numbers(tokens, context):
    """Replace bare reference numbers with the placeholder they stand for.

    In the corpus every identifier is normally a ``{{...}}`` slot, so a live
    query such as "where is my order 4471902" would otherwise become an
    out-of-vocabulary token and look out of scope. Working on the *token* stream
    rather than the raw string matters: the corpus also glues identifiers to the
    preceding word ("cancel purchase370795561790"), which only separates once
    the tokeniser has run.
    """
    slot = reference_slot(context)
    out = []
    for tok in tokens:
        digits = sum(1 for ch in tok if ch.isdigit())
        out.append(slot if (digits >= MIN_REF_DIGITS and RE_REF_NUMBER.match(tok))
                   else tok)
    return out


def preprocess_query(text, mapping):
    """Single entry point used by training *and* the live application.

    Applies the cleaning chain, then chat-speak normalisation and
    reference-number slotting, which only make sense on text typed by a customer.
    """
    cleaned = clean_text(text, mapping, normalise_chat=True)
    return slot_reference_numbers(tokenize(cleaned), cleaned)


def content_words(tokens):
    """Domain-bearing words: alphabetic, longer than two characters, not a stopword."""
    return [t for t in tokens if t.isalpha() and len(t) > 2 and t not in STOPWORDS]


def truncate_at_sentence(tokens, limit):
    """Cut a reply to ``limit`` tokens at the last complete sentence."""
    if len(tokens) <= limit:
        return tokens
    window = tokens[:limit]
    for i in range(len(window) - 1, -1, -1):
        if window[i] in (".", "!", "?"):
            return window[: i + 1]
    return window


def build_scope_lexicon(train_query_tokens):
    """Lexicon used by the out-of-scope gate at inference time (Section 2.2f)."""
    counter = Counter()
    for toks in train_query_tokens:
        counter.update(toks)
    known = sorted(t for t, c in counter.items() if c >= SCOPE_MIN_FREQ)
    keep = set(content_words(list(counter.keys())))
    content = Counter({t: c for t, c in counter.items() if t in keep})
    return {"query_vocab": known,
            "anchors": [t for t, _ in content.most_common(ANCHOR_TOP_K)],
            "stopwords": sorted(STOPWORDS), "min_freq": SCOPE_MIN_FREQ}


# ---- detokenisation: model output -> text shown to the customer ------------ #
_NO_SPACE_BEFORE = set(list(".,!?;:%)]}"))
_NO_SPACE_AFTER = set("([{$#")
_TIGHT_JOINERS = set("/@_")        # glued to the words on both sides
_PAIRED = set(['"', "'"])

SPECIAL_SURFACE = {
    "<url>": "{{Website URL}}", "<email>": "{{Customer Support Email}}",
    "<user>": "there", "<num>": "{{Number}}", "<unk>": "",
}


def _surface(tok, inverse):
    if tok in SPECIAL_SURFACE:
        return SPECIAL_SURFACE[tok]
    if tok == GENERIC_PLACEHOLDER:
        return "{{Details}}"
    if tok.startswith("<ph_") and tok.endswith(">"):
        return "{{" + inverse.get(tok, tok[4:-1].replace("_", " ").title()) + "}}"
    return tok


def detokenize(tokens, mapping=None):
    """Re-assembles decoder tokens into a readable, capitalised support reply."""
    mapping = mapping or {}
    inverse = {v: k for k, v in mapping.items() if v != GENERIC_PLACEHOLDER}
    words = [w for w in (_surface(t, inverse) for t in tokens) if w]

    pieces = []
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
            attach = open_quote[tok]        # a closing quote hugs the word
        elif prev in _PAIRED and open_quote.get(prev):
            attach = True                   # word follows an opening quote
        elif prev and prev[-1] in _NO_SPACE_AFTER and len(prev) == 1:
            attach = True
        elif tok.startswith("'") and len(tok) <= 3:
            attach = True                   # 's, 've, 'll, n't
        else:
            attach = False

        if tok in _PAIRED:
            open_quote[tok] = not open_quote[tok]
        pieces.append(tok if attach else " " + tok)

    out = "".join(pieces)
    out = re.sub(r"\bi\b", "I", out)
    out = re.sub(r"(^|(?<=[.!?]\s))([a-z])",
                 lambda m: m.group(1) + m.group(2).upper(), out)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    out = RE_MULTISPACE.sub(" ", out).strip()
    if out and out[-1] not in ".!?":
        out += "."
    return out


print("preprocessing functions defined")
''')

code(r'''
# =========================================================================== #
# Vocabulary  (Section 2.2e)
# =========================================================================== #
class Vocabulary(object):
    """Maps tokens <-> integer ids, with the four special tokens pinned to 0-3."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        assert self.stoi[PAD_TOKEN] == PAD_IDX
        assert self.stoi[SOS_TOKEN] == SOS_IDX
        assert self.stoi[EOS_TOKEN] == EOS_IDX
        assert self.stoi[UNK_TOKEN] == UNK_IDX

    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    @classmethod
    def build(cls, corpora, min_freq=MIN_FREQ, max_size=None):
        counter = Counter()
        for tokens in corpora:
            counter.update(tokens)
        kept = [(tok, c) for tok, c in counter.items() if c >= min_freq]
        kept.sort(key=lambda kv: (-kv[1], kv[0]))    # frequency desc, then a-z
        if max_size is not None:
            kept = kept[: max_size - len(SPECIAL_TOKENS)]
        vocab = cls(list(SPECIAL_TOKENS) + [tok for tok, _ in kept])
        vocab.freqs = dict(counter)
        return vocab

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"itos": self.itos}, fh, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh)["itos"])

    def encode(self, tokens, max_len, add_sos=True, add_eos=True):
        """Token list -> padded id list of exactly ``max_len`` entries."""
        budget = max_len - int(add_sos) - int(add_eos)
        ids = [self.stoi.get(t, UNK_IDX) for t in tokens[:budget]]
        if add_sos:
            ids = [SOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids + [PAD_IDX] * (max_len - len(ids))

    def decode(self, ids, strip_special=True):
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

    def coverage(self, counter):
        """Share of running-text tokens covered by the vocabulary."""
        total = sum(counter.values())
        if total == 0:
            return 0.0
        return sum(c for tok, c in counter.items() if tok in self.stoi) / float(total)


print("Vocabulary class defined")
''')

code(r'''
# =========================================================================== #
# The pipeline: raw CSV -> cleaned, de-duplicated, leakage-free splits
# =========================================================================== #
def group_aware_split(df, ratios=SPLIT_RATIOS, seed=RANDOM_SEED):
    """Split by *distinct query*, so no paraphrase of a test query leaks into train."""
    rng = np.random.RandomState(seed)
    keys = df["query_clean"].unique()
    rng.shuffle(keys)
    n_train = int(ratios[0] * len(keys))
    n_valid = int(ratios[1] * len(keys))
    assign = {}
    for i, key in enumerate(keys):
        assign[key] = ("train" if i < n_train else
                       "valid" if i < n_train + n_valid else "test")
    df = df.copy()
    df["split"] = df["query_clean"].map(assign)
    return df


def run_preprocessing(raw_csv=RAW_CSV, query_col="instruction",
                      response_col="response", min_freq=MIN_FREQ, verbose=True):
    """Executes steps (a)-(f) and writes every artefact into data/."""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    stats = {}
    log("[1/7] loading {}".format(os.path.basename(raw_csv)))
    df = pd.read_csv(raw_csv)
    df = df.rename(columns={query_col: "query", response_col: "response"})
    df = df[[c for c in ["query", "response", "category", "intent", "flags"]
             if c in df.columns]]
    stats["raw_pairs"] = int(len(df))
    log("      {} raw pairs".format(len(df)))

    log("[2/7] dropping empty / automated turns")
    before = len(df)
    df = df.dropna(subset=["query", "response"])
    stats["dropped_null"] = int(before - len(df))
    auto_mask = (df["query"].astype(str).str.contains(RE_AUTOMATED) |
                 df["response"].astype(str).str.contains(RE_AUTOMATED))
    stats["dropped_automated"] = int(auto_mask.sum())
    df = df[~auto_mask]
    log("      removed {} null / {} automated".format(
        stats["dropped_null"], stats["dropped_automated"]))

    log("[3/7] normalising placeholders and cleaning text")
    mapping = build_placeholder_map(list(df["query"]) + list(df["response"]))
    stats["placeholder_types"] = int(len(set(mapping.values())))
    # The query side goes through the *same* function the web app calls, so
    # training and inference can never drift apart.
    df["query_clean"] = [" ".join(preprocess_query(t, mapping)) for t in df["query"]]
    df["response_clean"] = [clean_text(t, mapping) for t in df["response"]]

    log("[4/7] removing duplicates and degenerate pairs")
    before = len(df)
    df = df.drop_duplicates(subset=["query_clean", "response_clean"])
    stats["dropped_duplicate_pairs"] = int(before - len(df))

    before = len(df)
    q_len = df["query_clean"].str.split().str.len()
    r_len = df["response_clean"].str.split().str.len()
    df = df[(q_len >= 2) & (q_len <= MAX_SRC_LEN - 2) & (r_len >= 3)]
    stats["dropped_too_short"] = int(before - len(df))

    # A single query may legitimately map to several agent phrasings; cap it so
    # the loss is not dominated by the most frequent intents.
    before = len(df)
    df = df.groupby("query_clean", group_keys=False, sort=False).head(4)
    stats["dropped_over_represented"] = int(before - len(df))
    stats["pairs_after_cleaning"] = int(len(df))
    log("      {} duplicate pairs, {} degenerate, {} over-represented -> {} kept".format(
        stats["dropped_duplicate_pairs"], stats["dropped_too_short"],
        stats["dropped_over_represented"], len(df)))

    log("[5/7] tokenising and truncating replies at a sentence boundary")
    df["query_tokens"] = [tokenize(t) for t in df["query_clean"]]
    raw_response_tokens = [tokenize(t) for t in df["response_clean"]]
    stats["truncated_responses"] = int(
        sum(1 for t in raw_response_tokens if len(t) > MAX_TGT_LEN - 2))
    df["response_tokens"] = [truncate_at_sentence(t, MAX_TGT_LEN - 2)
                             for t in raw_response_tokens]
    stats["mean_query_tokens"] = round(float(df["query_tokens"].apply(len).mean()), 2)
    stats["mean_response_tokens"] = round(float(df["response_tokens"].apply(len).mean()), 2)
    log("      mean query {} tok / mean response {} tok / {} responses truncated".format(
        stats["mean_query_tokens"], stats["mean_response_tokens"],
        stats["truncated_responses"]))

    log("[6/7] splitting {}/{}/{} by distinct query".format(*SPLIT_RATIOS))
    df = group_aware_split(df)
    counts = df["split"].value_counts().to_dict()
    stats["split_sizes"] = {k: int(v) for k, v in counts.items()}
    log("      " + " / ".join("{}: {}".format(k, v) for k, v in sorted(counts.items())))

    log("[7/7] building the vocabulary from the TRAIN split only")
    train_part = df[df["split"] == "train"]
    vocab = Vocabulary.build(
        list(train_part["query_tokens"]) + list(train_part["response_tokens"]),
        min_freq=min_freq)
    all_counter = Counter()
    for toks in list(df["query_tokens"]) + list(df["response_tokens"]):
        all_counter.update(toks)
    stats["vocab_size"] = len(vocab)
    stats["token_coverage"] = round(vocab.coverage(all_counter), 4)
    stats["min_freq"] = min_freq
    log("      vocab {} types, covers {:.2%} of all running tokens".format(
        len(vocab), stats["token_coverage"]))

    lexicon = build_scope_lexicon(list(train_part["query_tokens"]))
    stats["scope_query_vocab"] = len(lexicon["query_vocab"])
    log("      scope lexicon: {} known query words, {} domain anchors".format(
        len(lexicon["query_vocab"]), len(lexicon["anchors"])))

    # ---- persist ---------------------------------------------------------- #
    vocab.save(VOCAB_JSON)
    with open(PLACEHOLDER_JSON, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=1, ensure_ascii=False)
    with open(SCOPE_LEXICON_JSON, "w", encoding="utf-8") as fh:
        json.dump(lexicon, fh, indent=1)

    # Persist the *token* streams so training and the live app see identical text.
    df["query_clean"] = [" ".join(t) for t in df["query_tokens"]]
    df["response_clean"] = [" ".join(t) for t in df["response_tokens"]]
    out_cols = [c for c in ["query_clean", "response_clean", "category", "intent"]
                if c in df.columns]
    for name, path in (("train", TRAIN_CSV), ("valid", VALID_CSV), ("test", TEST_CSV)):
        df[df["split"] == name][out_cols].to_csv(path, index=False)
    with open(STATS_JSON, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    log("      wrote train.csv / valid.csv / test.csv and 4 JSON artefacts to data/")
    return df, vocab, mapping, lexicon, stats


artefacts = [TRAIN_CSV, VALID_CSV, TEST_CSV, VOCAB_JSON, PLACEHOLDER_JSON,
             SCOPE_LEXICON_JSON, STATS_JSON]
if REBUILD_DATA or not all(os.path.exists(p) for p in artefacts):
    t0 = time.time()
    clean_df, vocab, placeholder_map, lexicon, stats = run_preprocessing()
    print("\npreprocessing finished in {:.0f}s".format(time.time() - t0))
else:
    clean_df = None
    vocab = Vocabulary.load(VOCAB_JSON)
    placeholder_map = json.load(open(PLACEHOLDER_JSON))
    lexicon = json.load(open(SCOPE_LEXICON_JSON))
    stats = json.load(open(STATS_JSON))
    print("REBUILD_DATA is False - reusing the artefacts already in data/")
''')

md(r'''
### (a) Placeholder normalisation
''')

code(r'''
distinct_slots = sorted(set(placeholder_map.values()))
print("{} raw '{{{{...}}}}' strings -> {} vocabulary slots".format(
    len(placeholder_map), len(distinct_slots)))
print()
for slot in distinct_slots[:12]:
    print("   ", slot)
''')

md(r'''
### (b) Cleaning, normalisation and tokenisation
''')

code(r'''
demo = [
    "Heyyy @support!!! I NEED a refund ASAP \U0001F621\U0001F621 pls check "
    "https://shop.example.com/orders — thx!",
    "u guys nvr replied... my acct is locked, cant login \U0001F641 "
    "email me at jo.doe@mail.com",
    "<p>Where is my order 4471902?</p>",
]
for d in demo:
    print("raw    :", d)
    print("clean  :", clean_text(d, placeholder_map, normalise_chat=True))
    print("tokens :", preprocess_query(d, placeholder_map))
    print()
print("chat-speak table: {} entries, e.g.".format(len(CHAT_NORMALISATION)),
      {k: CHAT_NORMALISATION[k] for k in ["u", "pls", "asap", "recieve", "cant"]})
''')

md(r'''
### (c) The automated-message filter
''')

code(r'''
tests = ["This is an automated message, please do not reply to this email.",
         "Your ticket has been created automatically. Out of office until Monday.",
         "I understand you would like to cancel your order."]
for t in tests:
    print("{!r:70} -> automated: {}".format(t[:66], bool(RE_AUTOMATED.search(t))))
''')

md(r'''
### (d) Sentence-aware truncation
''')

code(r'''
lengths = raw["response"].head(200).str.split().str.len()
longest = raw["response"][lengths.idxmax()]
tokens = tokenize(clean_text(longest, placeholder_map))
cut = truncate_at_sentence(tokens, MAX_TGT_LEN - 2)
print("original : {} tokens".format(len(tokens)))
print("kept     : {} tokens (budget {})".format(len(cut), MAX_TGT_LEN - 2))
print("ends with: ... {}".format(" ".join(cut[-14:])))
print("\nfraction of the corpus affected: {}/{} replies".format(
    stats["truncated_responses"], stats["pairs_after_cleaning"]))
''')

md(r'''
### (e) Preprocessing summary, vocabulary and the splits
''')

code(r'''
print(json.dumps(stats, indent=2))
''')

code(r'''
print("vocabulary size :", len(vocab))
print("special tokens  :", vocab.itos[:4])
print("most frequent   :", vocab.itos[4:24])

train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)
test_df = pd.read_csv(TEST_CSV)
print("\nsplit sizes -> train {}, valid {}, test {}".format(
    len(train_df), len(valid_df), len(test_df)))

overlap = set(train_df["query_clean"]) & set(test_df["query_clean"])
print("queries shared between train and test:", len(overlap), "(group-aware split)")

ids = vocab.encode(train_df["query_clean"][0].split(), MAX_SRC_LEN)
print("\nencoded example:", train_df["query_clean"][0])
print("ids[:14]:", ids[:14], "... padded to", len(ids))
print("decoded  :", vocab.decode(ids))
''')

code(r'''
pd.DataFrame({
    "query":  [detokenize(q.split(), placeholder_map) for q in train_df["query_clean"][:5]],
    "reply":  [detokenize(r.split(), placeholder_map)[:180] + " ..."
               for r in train_df["response_clean"][:5]],
    "intent": train_df["intent"][:5],
})
''')

md(r'''
### (f) The out-of-scope lexicon
''')

code(r'''
print("full model vocabulary      : {} types".format(len(vocab)))
print("query-side scope lexicon   : {} types".format(len(lexicon["query_vocab"])))
print("stopwords excluded         : {}".format(len(lexicon["stopwords"])))
print("\ntop domain anchors:", ", ".join(lexicon["anchors"][:24]))
''')

# =========================================================================== #
# 3. Model
# =========================================================================== #
md(r'''
---
# 3. Model Development

## 3.1 Architecture

Two encoder–decoder models are defined below. Both expose the same interface, so
training, decoding and the app are architecture-agnostic:

```python
logits = model(src, tgt_in)             # teacher forcing, [B, T, V]
state  = model.init_decode(src)         # encode once
logits = model.decode_step(state, ys)   # next-token logits, [B, V]
state  = model.reorder_state(state, ix) # beam bookkeeping
```

**Transformer (default).** 3 encoder + 3 decoder layers, 4 attention heads,
`d_model` 256, feed-forward 512, pre-norm, sinusoidal positional encodings,
input/output embeddings tied. Attention is intrinsic: encoder self-attention,
decoder masked self-attention, and encoder–decoder cross-attention.

**LSTM + Bahdanau attention.** A 1-layer BiLSTM encoder (512 hidden per
direction), a 1-layer LSTM decoder with *input feeding*, and additive attention
`score(hᵈ, hᵉ) = vᵀ tanh(W_d hᵈ + W_e hᵉ)` over the encoder states.
''')

code(r'''
# =========================================================================== #
# Shared helpers
# =========================================================================== #
def make_pad_mask(seq):
    """True where the position is padding. [B, T]"""
    return seq.eq(PAD_IDX)


def causal_mask(size, device):
    """True above the diagonal, i.e. positions the decoder may not look at."""
    return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool),
                      diagonal=1)


class PositionalEncoding(nn.Module):
    """Classic fixed sinusoidal positions (Vaswani et al., 2017, Sec. 3.5)."""

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))       # [1, max_len, d_model]

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


# =========================================================================== #
# 1. Transformer encoder-decoder
# =========================================================================== #
class Seq2SeqTransformer(nn.Module):

    arch = "transformer"

    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_enc_layers=N_ENC_LAYERS, n_dec_layers=N_DEC_LAYERS,
                 ffn_dim=FFN_DIM, dropout=DROPOUT, max_len=512):
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
        self.generator.weight = self.embedding.weight     # weight tying
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, seq):
        return self.pos_encoding(self.embedding(seq) * math.sqrt(self.d_model))

    # -- training ----------------------------------------------------------- #
    def forward(self, src, tgt_in):
        src_kpm = make_pad_mask(src)
        out = self.transformer(
            self._embed(src), self._embed(tgt_in),
            tgt_mask=causal_mask(tgt_in.size(1), src.device),
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=make_pad_mask(tgt_in),
            memory_key_padding_mask=src_kpm,
        )
        return self.generator(out)

    # -- inference ---------------------------------------------------------- #
    def init_decode(self, src):
        src_kpm = make_pad_mask(src)
        memory = self.transformer.encoder(self._embed(src),
                                          src_key_padding_mask=src_kpm)
        return {"memory": memory, "src_kpm": src_kpm}

    def decode_step(self, state, ys):
        """Logits for the token that follows the prefix ``ys``. [B, V]"""
        out = self.transformer.decoder(
            self._embed(ys), state["memory"],
            tgt_mask=causal_mask(ys.size(1), ys.device),
            memory_key_padding_mask=state["src_kpm"],
        )
        return self.generator(out[:, -1])

    def reorder_state(self, state, index):
        return {"memory": state["memory"].index_select(0, index),
                "src_kpm": state["src_kpm"].index_select(0, index)}

    def expand_state(self, state, factor):
        return {"memory": state["memory"].repeat_interleave(factor, dim=0),
                "src_kpm": state["src_kpm"].repeat_interleave(factor, dim=0)}


# =========================================================================== #
# 2. LSTM encoder-decoder with Bahdanau attention
# =========================================================================== #
class BahdanauAttention(nn.Module):
    """score(h_dec, h_enc) = v^T tanh(W_dec h_dec + W_enc h_enc)."""

    def __init__(self, hidden_dim, enc_dim):
        super(BahdanauAttention, self).__init__()
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_enc = nn.Linear(enc_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_out, src_pad_mask):
        # dec_hidden [B, H] ; enc_out [B, S, E] ; src_pad_mask [B, S]
        scores = self.v(torch.tanh(
            self.W_dec(dec_hidden).unsqueeze(1) + self.W_enc(enc_out))).squeeze(-1)
        scores = scores.masked_fill(src_pad_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)                            # [B, S]
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)  # [B, E]
        return context, weights


class Seq2SeqLSTMAttention(nn.Module):

    arch = "lstm_attn"

    def __init__(self, vocab_size, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
                 n_layers=LSTM_LAYERS, dropout=DROPOUT):
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
        # input feeding: the previous context is concatenated to the next embedding
        self.decoder = nn.LSTM(emb_dim + enc_dim, hidden_dim, num_layers=n_layers,
                               batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden_dim + enc_dim, emb_dim)
        self.generator = nn.Linear(emb_dim, vocab_size)
        self.generator.weight = self.embedding.weight     # weight tying

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
        emb = self.dropout(self.embedding(token))                    # [B, 1, E]
        out, hidden = self.decoder(
            torch.cat([emb, context.unsqueeze(1)], dim=-1), hidden)
        dec_h = out.squeeze(1)
        context, attn = self.attention(dec_h, enc_out, src_pad_mask)
        feat = torch.tanh(self.out_proj(torch.cat([dec_h, context], dim=-1)))
        return self.generator(self.dropout(feat)), hidden, context, attn

    def forward(self, src, tgt_in):
        enc_out, hidden = self._encode(src)
        src_pad_mask = make_pad_mask(src)
        context = enc_out.new_zeros(src.size(0), enc_out.size(-1))
        logits = []
        for t in range(tgt_in.size(1)):
            step_logits, hidden, context, _ = self._step(
                tgt_in[:, t: t + 1], hidden, context, enc_out, src_pad_mask)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)                            # [B, T, V]

    def init_decode(self, src):
        enc_out, hidden = self._encode(src)
        return {"enc_out": enc_out, "src_kpm": make_pad_mask(src), "hidden": hidden,
                "context": enc_out.new_zeros(src.size(0), enc_out.size(-1)),
                "attn": None}

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


def build_model(arch, vocab_size):
    if arch == "transformer":
        return Seq2SeqTransformer(vocab_size,
                                  max_len=max(MAX_SRC_LEN, MAX_TGT_LEN) + 8)
    if arch == "lstm_attn":
        return Seq2SeqLSTMAttention(vocab_size)
    raise ValueError("unknown architecture: {}".format(arch))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("model classes defined")
''')

code(r'''
for arch in ["transformer", "lstm_attn"]:
    m = build_model(arch, len(vocab))
    print("{:<12} {:>10,} trainable parameters".format(arch, count_parameters(m)))

preview = build_model("transformer", len(vocab))
print()
print(preview.transformer.encoder.layers[0])
del preview, m
''')

md(r'''
## 3.2 Training

AdamW (lr 5e-4, β = 0.9/0.98, weight decay 1e-4), 400 warm-up steps then cosine
decay, label smoothing 0.1, gradient clipping at 1.0, batch size 64, early
stopping on validation loss (patience 4). Perplexity is computed from an
**unsmoothed** cross-entropy, so `exp(loss)` is a true perplexity.

One batching detail is worth noting: sequences are padded to a fixed length on
disk, but `collate` trims every batch to its own longest member, which roughly
halves the training time.

Training takes ~30 minutes on Apple MPS or a small GPU. `RETRAIN_MODEL` is
`False` by default, so the third cell below loads `data/best_model.pt` and the
recorded history instead; set it to `True` to train from scratch.
''')

code(r'''
# =========================================================================== #
# Dataset / DataLoader
# =========================================================================== #
class SupportPairDataset(Dataset):
    """Query/response pairs already cleaned and tokenised by the pipeline above."""

    def __init__(self, csv_path, vocab, max_src_len=MAX_SRC_LEN,
                 max_tgt_len=MAX_TGT_LEN):
        self.df = pd.read_csv(csv_path).fillna("")
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.queries = [str(s).split() for s in self.df["query_clean"]]
        self.responses = [str(s).split() for s in self.df["response_clean"]]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        src = self.vocab.encode(self.queries[idx], self.max_src_len)
        tgt = self.vocab.encode(self.responses[idx], self.max_tgt_len)
        return (torch.tensor(src, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long))


def collate(batch):
    """Stacks a batch and trims the padding that no example in it needs."""
    src = torch.stack([b[0] for b in batch])
    tgt = torch.stack([b[1] for b in batch])
    src = src[:, : int((src != PAD_IDX).sum(dim=1).max())]
    tgt = tgt[:, : int((tgt != PAD_IDX).sum(dim=1).max())]
    # teacher forcing: the decoder reads <sos> w1 .. wn-1 and predicts w1 .. <eos>
    return src, tgt[:, :-1].contiguous(), tgt[:, 1:].contiguous()


def make_loaders(vocab, batch_size=BATCH_SIZE):
    loaders = {}
    for name, path in (("train", TRAIN_CSV), ("valid", VALID_CSV), ("test", TEST_CSV)):
        loaders[name] = DataLoader(SupportPairDataset(path, vocab),
                                   batch_size=batch_size, shuffle=(name == "train"),
                                   collate_fn=collate, drop_last=False)
    return loaders


def pick_device(prefer="auto"):
    """MPS on Apple silicon, CUDA where available, CPU otherwise."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


device = pick_device()
loaders = make_loaders(vocab)
print("device:", device)
print("batches -> train {}, valid {}, test {}".format(
    len(loaders["train"]), len(loaders["valid"]), len(loaders["test"])))
''')

code(r'''
# =========================================================================== #
# Optimisation loop
# =========================================================================== #
def build_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warm-up followed by cosine decay to 10% of the peak LR."""
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step / float(warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1.0, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              clip=CLIP_NORM, limit_batches=None, log_every=50, tag=""):
    """One pass over ``loader``. Returns (mean token loss, tokens seen)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_tokens = 0.0, 0
    started = time.time()
    for step, (src, tgt_in, tgt_out) in enumerate(loader, start=1):
        if limit_batches and step > limit_batches:
            break
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        with torch.set_grad_enabled(training):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        n_tokens = int((tgt_out != PAD_IDX).sum())
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.detach()) * n_tokens
        total_tokens += n_tokens
        if training and step % log_every == 0:
            print("    {} step {:>4}/{}  loss {:.4f}  {:.0f} tok/s".format(
                tag, step, len(loader), total_loss / max(total_tokens, 1),
                total_tokens / max(time.time() - started, 1e-6)), flush=True)

    return total_loss / max(total_tokens, 1), total_tokens


def train_model(arch=ARCH, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE,
                limit_batches=None, device=None):
    """Trains, early-stops on validation loss and writes the best checkpoint."""
    device = device or pick_device()
    torch.manual_seed(RANDOM_SEED)
    model = build_model(arch, len(vocab)).to(device)

    train_config = {
        "arch": arch, "vocab_size": len(vocab), "device": str(device),
        "batch_size": BATCH_SIZE, "epochs": epochs, "lr": lr,
        "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
        "clip_norm": CLIP_NORM, "warmup_steps": WARMUP_STEPS, "patience": patience,
        "max_src_len": MAX_SRC_LEN, "max_tgt_len": MAX_TGT_LEN,
        "parameters": count_parameters(model),
        "train_pairs": len(loaders["train"].dataset),
        "valid_pairs": len(loaders["valid"].dataset),
        "test_pairs": len(loaders["test"].dataset),
    }
    if arch == "transformer":
        train_config.update({"d_model": D_MODEL, "n_heads": N_HEADS,
                             "enc_layers": N_ENC_LAYERS, "dec_layers": N_DEC_LAYERS,
                             "ffn_dim": FFN_DIM, "dropout": DROPOUT})
    else:
        train_config.update({"emb_dim": EMB_DIM, "hidden_dim": HIDDEN_DIM,
                             "lstm_layers": LSTM_LAYERS, "dropout": DROPOUT,
                             "attention": "bahdanau (additive)"})
    print("=" * 78)
    for k, v in train_config.items():
        print("  {:<18} {}".format(k, v))
    print("=" * 78, flush=True)

    # Smoothed loss for optimisation, plain CE for reporting perplexity.
    train_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX,
                                          label_smoothing=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
    steps_per_epoch = limit_batches or len(loaders["train"])
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, steps_per_epoch * epochs)

    history, best_valid, bad_epochs = [], float("inf"), 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _ = run_epoch(model, loaders["train"], train_criterion, device,
                                  optimizer=optimizer, scheduler=scheduler,
                                  limit_batches=limit_batches,
                                  tag="e{}".format(epoch))
        valid_loss, _ = run_epoch(model, loaders["valid"], eval_criterion, device,
                                  limit_batches=limit_batches)
        valid_ppl = math.exp(min(valid_loss, 20))
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "valid_loss": round(valid_loss, 4),
                        "valid_ppl": round(valid_ppl, 3),
                        "lr": round(scheduler.get_last_lr()[0], 6),
                        "seconds": round(time.time() - t0, 1)})
        print("epoch {:>2}/{}  train {:.4f}  valid {:.4f}  ppl {:.2f}  ({:.0f}s)".format(
            epoch, epochs, train_loss, valid_loss, valid_ppl, time.time() - t0),
            flush=True)

        if valid_loss < best_valid - 1e-4:
            best_valid, bad_epochs = valid_loss, 0
            torch.save({"model_state": model.state_dict(), "config": train_config,
                        "epoch": epoch, "valid_loss": valid_loss}, CHECKPOINT)
            print("           new best -> data/best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping: validation loss has not improved for "
                      "{} epochs.".format(patience))
                break
        with open(HISTORY_JSON, "w") as fh:
            json.dump({"config": train_config, "history": history}, fh, indent=2)

    with open(HISTORY_JSON, "w") as fh:
        json.dump({"config": train_config, "history": history}, fh, indent=2)

    # reload the best epoch, not the last one
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("\nBest validation loss {:.4f} (perplexity {:.2f})".format(
        best_valid, math.exp(min(best_valid, 20))))
    return model, history, train_config


print("training utilities defined")
''')

code(r'''
# --- train from scratch, or reuse the checkpoint already in data/ ----------- #
if RETRAIN_MODEL or not os.path.exists(CHECKPOINT):
    model, history, train_config = train_model(device=device)
else:
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    train_config = ckpt.get("config", {})
    if train_config.get("vocab_size", len(vocab)) != len(vocab):
        raise RuntimeError(
            "The checkpoint was trained with a {}-token vocabulary but the "
            "current one has {}. Set RETRAIN_MODEL = True.".format(
                train_config.get("vocab_size"), len(vocab)))
    model = build_model(train_config.get("arch", ARCH), len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    history = json.load(open(HISTORY_JSON))["history"]
    print("RETRAIN_MODEL is False - loaded data/best_model.pt")

print("\nTraining configuration")
for k, v in train_config.items():
    print("   {:<18} {}".format(k, v))
pd.DataFrame(history)
''')

code(r'''
import matplotlib
import matplotlib.pyplot as plt

epochs_ = [r["epoch"] for r in history]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(epochs_, [r["train_loss"] for r in history], "o-", label="train")
ax[0].plot(epochs_, [r["valid_loss"] for r in history], "s-", label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy per token")
ax[0].set_title("Training / validation loss"); ax[0].grid(alpha=.3); ax[0].legend()

ax[1].plot(epochs_, [r["valid_ppl"] for r in history], "s-", color="crimson")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("perplexity"); ax[1].set_yscale("log")
ax[1].set_title("Validation perplexity (log scale)"); ax[1].grid(alpha=.3)

fig.tight_layout()
fig.savefig(LOSS_CURVE_PNG, dpi=150)
plt.show()

best = min(history, key=lambda r: r["valid_loss"])
print("best epoch {}: valid loss {:.4f}, perplexity {:.2f}".format(
    best["epoch"], best["valid_loss"], best["valid_ppl"]))
print("curve saved to data/loss_curve.png")
''')

md(r'''
## 3.3 Decoding

Greedy decoding and beam search are both implemented below. Beam search ranks
hypotheses by a **length-normalised** score, `Σ log p / lengthᵃ` with `a = 0.7`
(Wu et al., 2016), so that short generic replies do not always win. A
**no-repeat-3-gram** constraint suppresses the classic seq2seq degeneration loop.

`ResponseGenerator` then wraps the model, the vocabulary, the placeholder map and
the scope lexicon into the single object used by the demonstrations here, by the
evaluation in Section 5 and by the Streamlit app — so what is measured is exactly
what a user sees.
''')

code(r'''
# =========================================================================== #
# Repetition control and the two decoders
# =========================================================================== #
def banned_by_no_repeat_ngram(sequence, n):
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


@torch.no_grad()
def greedy_decode(model, src, max_len=MAX_DECODE_LEN, no_repeat_ngram=NO_REPEAT_NGRAM):
    """Argmax decoding for a batch of queries. Returns (token ids, mean log-prob)."""
    device = src.device
    batch = src.size(0)
    state = model.init_decode(src)
    ys = torch.full((batch, 1), SOS_IDX, dtype=torch.long, device=device)

    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    logprob_sum = torch.zeros(batch, device=device)
    lengths = torch.zeros(batch, device=device)

    for _ in range(max_len - 1):
        logits = model.decode_step(state, ys)                        # [B, V]
        logits[:, PAD_IDX] = float("-inf")
        logits[:, UNK_IDX] = float("-inf")
        logits[:, SOS_IDX] = float("-inf")
        if no_repeat_ngram:
            for b in range(batch):
                for tok in banned_by_no_repeat_ngram(ys[b].tolist(), no_repeat_ngram):
                    logits[b, tok] = float("-inf")

        best_lp, best = F.log_softmax(logits, dim=-1).max(dim=-1)
        active = ~finished
        logprob_sum += best_lp * active
        lengths += active.float()
        best = torch.where(finished, torch.full_like(best, PAD_IDX), best)
        ys = torch.cat([ys, best.unsqueeze(1)], dim=1)
        finished = finished | best.eq(EOS_IDX)
        if bool(finished.all()):
            break

    outputs, scores = [], []
    for b in range(batch):
        seq = [int(t) for t in ys[b, 1:].tolist()]
        if EOS_IDX in seq:
            seq = seq[: seq.index(EOS_IDX)]
        outputs.append(seq)
        scores.append(float(logprob_sum[b] / max(float(lengths[b]), 1.0)))
    return outputs, scores


@torch.no_grad()
def beam_search_decode(model, src, beam_size=BEAM_SIZE, max_len=MAX_DECODE_LEN,
                       length_penalty=LENGTH_PENALTY,
                       no_repeat_ngram=NO_REPEAT_NGRAM):
    """Beam search for a **single** query (src is [1, S])."""
    device = src.device
    state = model.expand_state(model.init_decode(src), beam_size)
    ys = torch.full((beam_size, 1), SOS_IDX, dtype=torch.long, device=device)

    scores = torch.full((beam_size,), float("-inf"), device=device)
    scores[0] = 0.0                        # only the first beam is live at t=0
    finished = []

    for _ in range(max_len - 1):
        logits = model.decode_step(state, ys)
        logits[:, PAD_IDX] = float("-inf")
        logits[:, UNK_IDX] = float("-inf")
        logits[:, SOS_IDX] = float("-inf")
        if no_repeat_ngram:
            for b in range(ys.size(0)):
                for tok in banned_by_no_repeat_ngram(ys[b].tolist(), no_repeat_ngram):
                    logits[b, tok] = float("-inf")

        logprobs = F.log_softmax(logits, dim=-1)
        top_scores, top_ix = (scores.unsqueeze(1) + logprobs).view(-1).topk(beam_size)
        beam_ix = torch.div(top_ix, logprobs.size(-1), rounding_mode="floor")
        token_ix = top_ix % logprobs.size(-1)

        ys = torch.cat([ys.index_select(0, beam_ix), token_ix.unsqueeze(1)], dim=1)
        state = model.reorder_state(state, beam_ix)
        scores = top_scores

        # Retire any beam that produced <eos>; keep the slot alive but dead.
        for b in range(beam_size):
            if int(token_ix[b]) == EOS_IDX:
                seq = [int(t) for t in ys[b, 1:-1].tolist()]
                finished.append((float(scores[b]), seq, len(seq) + 1))
                scores[b] = float("-inf")
        if len(finished) >= beam_size or bool(torch.isinf(scores).all()):
            break

    if not finished:          # hit max_len without any beam emitting <eos>
        for b in range(beam_size):
            if not torch.isinf(scores[b]):
                seq = [int(t) for t in ys[b, 1:].tolist()]
                finished.append((float(scores[b]), seq, len(seq)))

    best_total, best_seq, best_len = max(
        finished, key=lambda item: item[0] / (max(item[2], 1) ** length_penalty))
    return best_seq, best_total / max(best_len, 1)                   # mean log-prob


print("decoders defined")
''')

code(r'''
# =========================================================================== #
# End-to-end generator: query text -> support reply, with the scope gate
# =========================================================================== #
class ResponseGenerator(object):

    def __init__(self, model, vocab, placeholder_map, scope_lexicon, device=None):
        self.model = model
        self.vocab = vocab
        self.placeholder_map = placeholder_map
        self.query_vocab = set(scope_lexicon["query_vocab"])
        self.anchors = set(scope_lexicon["anchors"])
        self.device = device or next(model.parameters()).device
        self.train_config = {}
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint=CHECKPOINT, device=None):
        """Rebuild the generator from data/ alone - this is what app.py does."""
        device = device or pick_device()
        vocab_ = Vocabulary.load(VOCAB_JSON)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model_ = build_model(ckpt.get("config", {}).get("arch", ARCH),
                             len(vocab_)).to(device)
        model_.load_state_dict(ckpt["model_state"])
        gen = cls(model_, vocab_, json.load(open(PLACEHOLDER_JSON)),
                  json.load(open(SCOPE_LEXICON_JSON)), device)
        gen.train_config = ckpt.get("config", {})
        return gen

    # ------------------------------------------------------------------ #
    def _encode_query(self, text):
        tokens = preprocess_query(text, self.placeholder_map)
        ids = self.vocab.encode(tokens, MAX_SRC_LEN)
        return tokens, torch.tensor([ids], dtype=torch.long, device=self.device)

    def _scope_check(self, tokens, avg_logprob):
        """Two cheap, explainable signals decide whether we answer at all:

        1. the share of *content* words the training queries have never seen;
        2. the decoder's own mean per-token log-probability.
        """
        content = content_words(tokens)
        if content:
            unknown = [t for t in content if t not in self.query_vocab]
            oov = len(unknown) / float(len(content))
        else:
            # only stopwords and punctuation: nothing to ground an answer on,
            # unless the query still holds a known slot such as an order number
            unknown, oov = [], (0.0 if any(t.startswith("<ph_") for t in tokens) else 1.0)

        n_anchors = sum(1 for t in content if t in self.anchors)
        in_scope = (oov <= OOS_OOV_RATIO) and (avg_logprob >= OOS_MIN_AVG_LOGPROB)

        if oov > OOS_OOV_RATIO:
            reason = ("{:.0%} of the meaningful words in this query never appear "
                      "in the support corpus ({})".format(
                          oov, ", ".join(unknown[:5]) or "none"))
        elif avg_logprob < OOS_MIN_AVG_LOGPROB:
            reason = ("the decoder is not confident (mean log-probability "
                      "{:.2f})".format(avg_logprob))
        else:
            reason = "in scope ({} known support term{})".format(
                n_anchors, "" if n_anchors == 1 else "s")

        return {"in_scope": in_scope, "oov_ratio": round(oov, 3),
                "n_anchors": n_anchors, "unknown_words": unknown[:8],
                "avg_logprob": round(avg_logprob, 3), "reason": reason}

    # ------------------------------------------------------------------ #
    def generate(self, text, strategy=DECODE_STRATEGY, beam_size=BEAM_SIZE,
                 max_len=MAX_DECODE_LEN, apply_scope_check=True):
        """Returns the reply plus every diagnostic the report and the UI display."""
        tokens, src = self._encode_query(text)
        if not tokens:
            return {"query": text, "response": OOS_MESSAGE, "query_tokens": [],
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
                    if (scope["in_scope"] or not apply_scope_check) else OOS_MESSAGE)

        result = {"query": text, "query_tokens": tokens, "response": response,
                  "raw_tokens": out_tokens, "strategy": strategy,
                  "beam_size": beam_size if strategy == "beam" else None}
        result.update(scope)
        return result

    def generate_batch(self, texts, strategy="greedy", max_len=MAX_DECODE_LEN,
                       apply_scope_check=True, batch_size=32):
        """File-upload path: greedy decoding in batches; beam falls back to a loop."""
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
                rows.append(self.vocab.encode(toks, MAX_SRC_LEN))
            src = torch.tensor(rows, dtype=torch.long, device=self.device)
            ids_batch, lp_batch = greedy_decode(self.model, src, max_len=max_len)

            for text, toks, ids, lp in zip(chunk, token_lists, ids_batch, lp_batch):
                out_tokens = self.vocab.decode(ids)
                scope = self._scope_check(toks, lp)
                response = (detokenize(out_tokens, self.placeholder_map)
                            if (scope["in_scope"] or not apply_scope_check)
                            else OOS_MESSAGE)
                row = {"query": text, "query_tokens": toks, "response": response,
                       "raw_tokens": out_tokens, "strategy": "greedy",
                       "beam_size": None}
                row.update(scope)
                results.append(row)
        return results


gen = ResponseGenerator(model, vocab, placeholder_map, lexicon, device)
gen.train_config = train_config
print("generator ready:", train_config.get("arch"), "|",
      "{:,} parameters".format(train_config.get("parameters", count_parameters(model))),
      "| device:", device)
''')

code(r'''
demo_queries = [
    "i want to cancel order 4471902",
    "how do i get a refund for my last purchase?",
    "i forgot my password and cannot sign in",
    "where is my package? it has not arrived yet",
]
for q in demo_queries:
    g = gen.generate(q, strategy="greedy", apply_scope_check=False)
    b = gen.generate(q, strategy="beam", beam_size=3, apply_scope_check=False)
    print("=" * 100)
    print("QUERY :", q)
    print("GREEDY:", textwrap.fill(g["response"], 96, subsequent_indent="        "))
    print("BEAM  :", textwrap.fill(b["response"], 96, subsequent_indent="        "))
    print("        [greedy logprob {:.2f} | beam logprob {:.2f}]".format(
        g["avg_logprob"], b["avg_logprob"]))
''')

# =========================================================================== #
# 4. Application
# =========================================================================== #
md(r'''
---
# 4. Application Development

`app.py` is a Streamlit application launched from the project folder with:

```bash
streamlit run app.py     #  ->  http://localhost:8501
```

It rebuilds the same generator from `data/best_model.pt` (the equivalent of
`ResponseGenerator.from_checkpoint()` defined above), so the app and this
notebook answer identically, and adds the user interface:

**Chat tab** — a chat-style interface (`st.chat_message` / `st.chat_input`). The
transcript is kept in `st.session_state` and re-rendered on every run, so the
whole conversation stays on screen. Every reply carries a *Diagnostics* panel
(decoding strategy, mean log-probability, unknown-word ratio, in/out-of-scope
decision, latency), and the transcript can be downloaded.

**Batch / file-upload tab** — accepts `.txt` (one query per line) or `.csv`
(the user picks the column). Every row is answered with batched greedy decoding,
results are shown in a table with the scope flag and confidence, and the whole
set is downloadable as CSV.

**Sidebar** — checkpoint selection, greedy vs beam, beam size, maximum reply
length, an on/off switch for out-of-scope refusal, a diagnostics toggle and a
*clear conversation* button.

## 4.1 Screenshots
''')

code(r'''
from IPython.display import Image, display, Markdown

shots = sorted(glob.glob(os.path.join(SCREENSHOT_DIR, "*.png")))
if not shots:
    display(Markdown("> **No screenshots found.** Start the app with "
                     "`streamlit run app.py` and save PNGs into "
                     "`data/screenshots/`."))
for path in shots:
    display(Markdown("**{}**".format(os.path.basename(path))))
    display(Image(filename=path, width=1000))
''')

md(r'''
## 4.2 The batch path, executed here

The batch tab calls `generate_batch` exactly as the cell below does, on the
sample query file shipped in `data/`.
''')

code(r'''
with open(SAMPLE_QUERIES_TXT) as fh:
    batch_queries = [ln.strip() for ln in fh if ln.strip()]

batch_results = gen.generate_batch(batch_queries, strategy="greedy",
                                   apply_scope_check=True)
pd.DataFrame([{
    "query": r["query"],
    "generated_response": r["response"][:150] + ("..." if len(r["response"]) > 150 else ""),
    "in_scope": r["in_scope"],
    "oov": r["oov_ratio"],
    "logprob": r["avg_logprob"],
} for r in batch_results])
''')

# =========================================================================== #
# 5. Evaluation
# =========================================================================== #
md(r'''
---
# 5. Evaluation and Demonstration

## 5.1 Automatic metrics

Computed on the held-out test split:

* **BLEU** and **chrF** (sacreBLEU) — n-gram overlap with the reference reply.
* **ROUGE-1 / ROUGE-2 / ROUGE-L** F1 — recall-oriented overlap, the more
  informative family for long replies.
* **Perplexity** — `exp` of the mean token cross-entropy of the *reference*
  replies under the model; measures the language model itself, independent of the
  decoding strategy.
* **distinct-1 / distinct-2** and the repeated-reply rate — these expose the
  "safe generic answer" failure mode that seq2seq chatbots are notorious for.
* The **out-of-scope gate** on in-domain versus deliberately out-of-domain queries.

The evaluation needs two extra packages (`pip install sacrebleu rouge-score`) and
takes about five minutes for 400 queries with both decoding strategies. It runs
automatically when `data/metrics.json` is absent and reuses that file otherwise —
set `RECOMPUTE_METRICS = True` to force it.
''')

code(r'''
# =========================================================================== #
# Metric helpers
# =========================================================================== #
OUT_OF_DOMAIN_QUERIES = [
    "what is the boiling point of water on mars",
    "write me a python function that sorts a list",
    "who won the football world cup in 1998",
    "can you tell me a joke about penguins",
    "what should i cook for dinner tonight",
    "explain the theory of general relativity",
]


@torch.no_grad()
def corpus_perplexity(model, csv_path, vocab, device, batch_size=32):
    """exp(mean token cross-entropy) of the *reference* replies."""
    loader = DataLoader(SupportPairDataset(csv_path, vocab), batch_size=batch_size,
                        shuffle=False, collate_fn=collate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction="sum")
    total_loss, total_tokens = 0.0, 0
    model.eval()
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        total_loss += float(criterion(logits.reshape(-1, logits.size(-1)),
                                      tgt_out.reshape(-1)))
        total_tokens += int((tgt_out != PAD_IDX).sum())
    return math.exp(total_loss / max(total_tokens, 1))


def distinct_n(token_lists, n):
    grams = Counter()
    for toks in token_lists:
        for i in range(len(toks) - n + 1):
            grams[tuple(toks[i: i + n])] += 1
    total = sum(grams.values())
    return len(grams) / float(total) if total else 0.0


def generic_rate(responses, top_k=5):
    """How often the model falls back on its handful of favourite replies."""
    counts = Counter(r.strip().lower() for r in responses)
    repeated = sum(c for _, c in counts.items() if c > 1)
    return {"unique_responses": len(counts),
            "repeated_response_rate": round(repeated / max(len(responses), 1), 4),
            "most_common": [{"count": c, "response": r[:160]}
                            for r, c in counts.most_common(top_k)]}


def run_evaluation(gen, limit=EVAL_LIMIT, strategies=("greedy", "beam"),
                   sample_size=20):
    """Full metric suite; writes data/metrics.json and the manual rating sheet."""
    import sacrebleu
    from rouge_score import rouge_scorer

    random.seed(RANDOM_SEED)
    df = pd.read_csv(TEST_CSV).fillna("")
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=RANDOM_SEED)
    df = df.reset_index(drop=True)
    queries = [str(q) for q in df["query_clean"]]
    references = [str(r) for r in df["response_clean"]]
    print("evaluating on {} held-out queries".format(len(queries)), flush=True)

    metrics = {"checkpoint": os.path.basename(CHECKPOINT),
               "arch": gen.train_config.get("arch"),
               "n_test_queries": len(queries),
               "train_config": gen.train_config}

    print("[1/4] perplexity on the full valid/test splits ...", flush=True)
    metrics["perplexity_test"] = round(
        corpus_perplexity(gen.model, TEST_CSV, gen.vocab, gen.device), 3)
    metrics["perplexity_valid"] = round(
        corpus_perplexity(gen.model, VALID_CSV, gen.vocab, gen.device), 3)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per_strategy, decoded_cache = {}, {}
    for si, strategy in enumerate(strategies, start=2):
        print("[{}/4] decoding with {} ...".format(si, strategy), flush=True)
        if strategy == "greedy":
            results = gen.generate_batch(queries, strategy="greedy",
                                         apply_scope_check=False)
        else:
            results = []
            for i, q in enumerate(queries):
                results.append(gen.generate(q, strategy="beam",
                                            apply_scope_check=False))
                if (i + 1) % 100 == 0:
                    print("      {}/{}".format(i + 1, len(queries)), flush=True)

        hypotheses = [" ".join(r["raw_tokens"]) for r in results]
        decoded_cache[strategy] = results

        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a",
                                     lowercase=True)
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
            per_strategy[strategy]["distinct_2"]), flush=True)
    metrics["by_strategy"] = per_strategy

    print("[4/4] out-of-scope gate ...", flush=True)
    in_domain = gen.generate_batch(queries[:200], strategy="greedy",
                                   apply_scope_check=True)
    ood = [gen.generate(q, strategy="beam", apply_scope_check=True)
           for q in OUT_OF_DOMAIN_QUERIES]
    metrics["scope_gate"] = {
        "in_domain_accepted": round(
            sum(1 for r in in_domain if r["in_scope"]) / max(len(in_domain), 1), 4),
        "out_of_domain_rejected": round(
            sum(1 for r in ood if not r["in_scope"]) / max(len(ood), 1), 4),
        "thresholds": {"oov_ratio": OOS_OOV_RATIO,
                       "min_avg_logprob": OOS_MIN_AVG_LOGPROB},
        "out_of_domain_detail": [
            {"query": r["query"], "in_scope": r["in_scope"],
             "oov_ratio": r["oov_ratio"], "avg_logprob": r["avg_logprob"],
             "reason": r["reason"]} for r in ood],
    }

    # ---- manual rating sheet + qualitative sample ------------------------- #
    primary = strategies[-1]
    idx = random.sample(range(len(queries)), min(sample_size, len(queries)))
    rows = []
    for i in idx:
        r = decoded_cache[primary][i]
        rows.append({
            "query": detokenize(queries[i].split(), gen.placeholder_map),
            "reference_reply": detokenize(references[i].split(), gen.placeholder_map),
            "generated_reply": detokenize(r["raw_tokens"], gen.placeholder_map),
            "intent": df.get("intent", pd.Series([""] * len(df)))[i],
            "relevance_1_to_5": "", "fluency_1_to_5": "",
            "would_send_as_is_yes_no": "", "rater_comment": "",
        })
    pd.DataFrame(rows).to_csv(RATING_SHEET_CSV, index=False)
    metrics["qualitative_samples"] = [
        {"query": r["query"], "reference": r["reference_reply"],
         "generated": r["generated_reply"]} for r in rows[:8]]

    with open(METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print("\nmetrics -> data/metrics.json | rating sheet -> data/manual_rating_sheet.csv")
    return metrics


if RECOMPUTE_METRICS or not os.path.exists(METRICS_JSON):
    t0 = time.time()
    metrics = run_evaluation(gen)
    print("evaluation finished in {:.0f}s".format(time.time() - t0))
else:
    metrics = json.load(open(METRICS_JSON))
    print("RECOMPUTE_METRICS is False - reusing data/metrics.json")
''')

code(r'''
print("architecture      :", metrics["arch"])
print("test queries      :", metrics["n_test_queries"])
print("perplexity  valid :", metrics["perplexity_valid"])
print("perplexity  test  :", metrics["perplexity_test"])

rows = []
for strategy, m in metrics["by_strategy"].items():
    rows.append({
        "decoding": strategy,
        "BLEU": m["bleu_corpus"], "chrF": m["chrf"],
        "ROUGE-1": m["rouge1_f"], "ROUGE-2": m["rouge2_f"], "ROUGE-L": m["rougeL_f"],
        "distinct-1": m["distinct_1"], "distinct-2": m["distinct_2"],
        "unique replies": m["genericity"]["unique_responses"],
        "mean len (hyp/ref)": "{} / {}".format(m["mean_hypothesis_tokens"],
                                               m["mean_reference_tokens"]),
    })
pd.DataFrame(rows).set_index("decoding")
''')

md(r'''
## 5.2 Manual relevance rating

The evaluation writes `data/manual_rating_sheet.csv`: a random sample of test
queries with the reference and the generated reply, and empty columns for
**relevance (1–5)**, **fluency (1–5)** and **would-send-as-is (yes/no)**. Fill it
in by hand and report the averages in the write-up.
''')

code(r'''
if os.path.exists(RATING_SHEET_CSV):
    sheet = pd.read_csv(RATING_SHEET_CSV)
    print("{} rows for manual rating -> data/manual_rating_sheet.csv".format(len(sheet)))
    display(sheet[["query", "generated_reply"]].head(5))
else:
    print("No rating sheet yet - set RECOMPUTE_METRICS = True and re-run the "
          "evaluation cell above to write data/manual_rating_sheet.csv.")
''')

md(r'''
## 5.3 Demonstration, including an out-of-domain query

The last three queries below are deliberately outside the support domain.
''')

code(r'''
demo = [
    "i want to cancel my order",
    "how long does a refund take to appear on my card?",
    "i cannot log into my account, i forgot my password",
    "i need to speak to a human agent",
    # ---- out of domain ----
    "what is the boiling point of water on mars",
    "write me a python function that sorts a list",
    "who won the football world cup in 1998",
]
for q in demo:
    r = gen.generate(q, strategy="beam", beam_size=3, apply_scope_check=True)
    print("=" * 100)
    print("QUERY   :", q)
    print("REPLY   :", textwrap.fill(r["response"], 96, subsequent_indent="          "))
    print("SCOPE   : in_scope={}  oov={:.2f}  mean_logprob={:.2f}  ({})".format(
        r["in_scope"], r["oov_ratio"], r["avg_logprob"], r["reason"]))
''')

code(r'''
gate = metrics["scope_gate"]
print("thresholds:", gate["thresholds"])
print("in-domain queries accepted   : {:.1%}".format(gate["in_domain_accepted"]))
print("out-of-domain queries refused: {:.1%}".format(gate["out_of_domain_rejected"]))
pd.DataFrame(gate["out_of_domain_detail"])
''')

md(r'''
### What an out-of-domain query produces **without** the gate

This is the argument for having the gate at all: the decoder never abstains on
its own — it always emits fluent, confident, and wrong support language.
''')

code(r'''
for q in ["what is the boiling point of water on mars",
          "who won the football world cup in 1998"]:
    r = gen.generate(q, strategy="beam", apply_scope_check=False)
    print("QUERY :", q)
    print("RAW   :", textwrap.fill(r["response"], 96, subsequent_indent="        "))
    print()
''')

# =========================================================================== #
# 6. Observations
# =========================================================================== #
md(r'''
---
# 6. Observations, Conclusion and References

## 6.1 Observations

**Generic replies.** The model converges on a small set of high-frequency
openings — *"I understand …"*, *"I'm sorry to hear …"*, *"Thank you for reaching
out …"*. This is the well-documented safe-answer bias of maximum-likelihood
seq2seq models (Li et al., 2016): under cross-entropy, the lowest-risk output is
the most frequent one. The `distinct-1/2` figures and the repeated-reply count in
§5.1 quantify it. Beam search makes it *worse* than greedy decoding unless
length normalisation is applied, because short, common replies accumulate less
negative log-probability.

**Repetition.** Without a constraint, the decoder loops on politeness clauses
("*I'm here to help you. I'm here to help you.*"). The no-repeat-3-gram rule in
§3.3 removes almost all of it; the remaining repetition is inter-sentential
(the same idea rephrased), which n-gram blocking cannot catch.

**Factual reliability.** The model has **no access to any real system**. When it
emits `{{Order Number}}` or `{{Customer Support Hours}}` it is reproducing a slot
it saw in training, not looking anything up. Any number, date or policy detail in
a generated reply is a *pattern*, not a fact. The placeholders are therefore a
feature: they mark exactly where a human must supply real data.

**Safety of automated customer communication.** Three risks matter in production:
1. *Confident nonsense on out-of-scope input* — mitigated by the scope gate, which
   abstains rather than inventing.
2. *Commitments the business cannot keep* — a drafted "your refund will be
   processed in 3–5 days" is a promise. Drafts must be reviewed before sending.
3. *Tone on distressed contacts* — the corpus is synthetic and uniformly polite;
   it contains no genuinely angry customers, so the model is untested on them.

The recommended deployment is therefore **agent-assist, not auto-send**: the
model proposes, an agent approves, and the scope gate routes anything unusual to
a human.

**Data limitations.** The corpus is hybrid-synthetic and highly templated. That
makes it learnable by a 5 M-parameter model trained on a laptop, but it inflates
overlap metrics relative to what real, messy support logs would give. The numbers
in §5.1 should be read as an upper bound.

## 6.2 Conclusion

An encoder–decoder response generator was built end to end: 26,872 support pairs
were cleaned, normalised, de-duplicated and split without leakage; a Transformer
seq2seq was trained from scratch (an LSTM + Bahdanau attention variant is defined
in §3.1 and selectable by setting `ARCH = "lstm_attn"`, but every number reported
above is the Transformer's); greedy and length-normalised beam decoding were
implemented with repetition control; and the model was served through a Streamlit
chat application with a batch upload mode, per-reply diagnostics and an explicit
out-of-scope refusal path. The system drafts fluent, intent-appropriate support
replies for the eleven covered categories and abstains on anything else.

The main limitation is genericity — an artefact of maximum-likelihood training on
a templated corpus. Natural extensions: multi-turn context in the encoder,
sub-word (BPE) vocabulary to remove `<unk>` entirely, a copy/pointer mechanism for
order and invoice numbers, an explicit intent classifier as a second scope
signal, and fine-tuning a pre-trained seq2seq (T5/BART) as a stronger baseline.

## 6.3 References

1. Sutskever, I., Vinyals, O., Le, Q. *Sequence to Sequence Learning with Neural Networks.* NeurIPS 2014.
2. Bahdanau, D., Cho, K., Bengio, Y. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
3. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
4. Wu, Y. et al. *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation.* arXiv:1609.08144, 2016.
5. Li, J. et al. *A Diversity-Promoting Objective Function for Neural Conversation Models.* NAACL 2016.
6. Papineni, K. et al. *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL 2002.
7. Lin, C.-Y. *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop 2004.
8. Post, M. *A Call for Clarity in Reporting BLEU Scores.* WMT 2018 (sacreBLEU).
9. Bitext. *Customer Service Tagged Training Dataset for LLM-based Virtual Assistants.* Hugging Face, 2024. Licence CDLA-Sharing-1.0.
10. Streamlit Documentation — Chat elements. <https://docs.streamlit.io/develop/api-reference/chat>
''')


# =========================================================================== #
# Assemble the .ipynb
# =========================================================================== #
def to_source(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main():
    cells = []
    for i, (kind, text) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {}, "id": "cell{:03d}".format(i),
                "source": to_source(text)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print("wrote {} ({} cells: {} code, {} markdown)".format(
        os.path.relpath(OUT, ROOT), len(CELLS), n_code, len(CELLS) - n_code))


if __name__ == "__main__":
    main()
