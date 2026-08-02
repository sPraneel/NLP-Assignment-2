"""Generate notebooks/customer_support_response_generation.ipynb.

The notebook is the submission artefact: it walks through Tasks 1-6 and, when
executed, embeds real outputs (statistics, loss curves, generated replies,
metrics). It re-uses the artefacts produced by ``src/*.py`` rather than
retraining, so a full execution takes about two minutes.

    python scripts/build_notebook.py
    jupyter nbconvert --execute --to notebook --inplace \
        notebooks/customer_support_response_generation.ipynb
"""

import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "customer_support_response_generation.ipynb")

GROUP_TABLE = """
| Name | BITS ID | Contribution |
|---|---|---|
| _&lt;member 1&gt;_ | _&lt;2xxxxxxx&gt;_ | 25% |
| _&lt;member 2&gt;_ | _&lt;2xxxxxxx&gt;_ | 25% |
| _&lt;member 3&gt;_ | _&lt;2xxxxxxx&gt;_ | 25% |
| _&lt;member 4&gt;_ | _&lt;2xxxxxxx&gt;_ | 25% |
"""

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------- #
md("""
# Customer Support Response Generation Chatbot

**M.Tech. AIML — Natural Language Processing (S2-25_AIMLCZG530)**
Assignment 2 · GS-3 · Encoder–Decoder Conversational Response Generation

### Group details
""" + GROUP_TABLE + """
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

Everything below runs against the artefacts produced by the scripts in `src/`.
""")

code("""
import json, os, sys, textwrap, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import pandas as pd, torch
import config as C

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

print("project root :", ROOT)
print("python       :", sys.version.split()[0])
print("torch        :", torch.__version__)
""")

# --------------------------------------------------------------------------- #
md("""
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
locally with `streamlit run app/app.py`; the model is < 10 M parameters so it
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
""")

# --------------------------------------------------------------------------- #
md("""
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
pairs, 10 k–1 L rows, support-desk messages, with optional intent metadata.
""")

code("""
raw = pd.read_csv(C.RAW_CSV)
print("shape:", raw.shape)
print("columns:", list(raw.columns))
raw.head(3)
""")

code("""
summary = pd.DataFrame({
    "customer query (words)": raw["instruction"].str.split().str.len().describe(),
    "agent reply (words)":    raw["response"].str.split().str.len().describe(),
}).round(2)
print(summary.to_string())
print("\\ndistinct intents   :", raw["intent"].nunique())
print("distinct categories:", raw["category"].nunique())
print()
print(raw["category"].value_counts().to_string())
""")

md("""
The `flags` column encodes the linguistic variation deliberately injected into
each query — `Q` colloquial, `Z` noise/typos, `L` offensive-free paraphrase,
`M` morphological variation, and so on. That is useful: the encoder sees
misspelled and informal phrasings of the same intent, which is what a real
support inbox looks like.
""")

code("""
print(raw["flags"].value_counts().head(8).to_string())
for i in [0, 2, 7]:
    print("\\n--- row", i, "| intent:", raw['intent'][i], "| flags:", raw['flags'][i])
    print("Q:", raw["instruction"][i])
    print("A:", textwrap.fill(raw["response"][i][:400], 100))
""")

md("""
## 2.2 Preprocessing

`src/preprocess.py` implements the full chain. The cells below demonstrate each
step on real rows.

**a) Placeholder normalisation.** The corpus marks variable data as
`{{Order Number}}`, `{{Account Type}}`, … Left alone, the braces would be
shredded by tokenisation. Each of the 43 frequent slots becomes **one**
vocabulary token; the long tail collapses into `<ph_details>`. The mapping is
saved so replies can be rendered back with readable placeholders.
""")

code("""
from preprocess import (build_placeholder_map, clean_text, tokenize,
                        preprocess_query, detokenize, truncate_at_sentence,
                        CHAT_NORMALISATION, RE_AUTOMATED)

placeholder_map = json.load(open(os.path.join(C.PROCESSED_DIR, "placeholder_map.json")))
distinct_slots = sorted(set(placeholder_map.values()))
print("{} raw '{{{{...}}}}' strings -> {} vocabulary slots".format(
    len(placeholder_map), len(distinct_slots)))
print()
for slot in distinct_slots[:12]:
    print("   ", slot)
""")

md("""
**b) Cleaning and normalisation.** HTML tags, URLs, e-mail addresses,
`@usernames`, hashtags, emoji and zero-width characters are removed or replaced
by typed tokens; curly quotes/dashes are normalised; repeated characters
(`heyyyy`) and punctuation (`???`) are collapsed; everything is lower-cased.
Chat-speak is expanded on the query side only.
""")

code("""
demo = [
    "Heyyy @support!!! I NEED a refund ASAP 😡😡 pls check https://shop.example.com/orders — thx!",
    "u guys nvr replied... my acct is locked, cant login 🙁 email me at jo.doe@mail.com",
    "<p>Where is my order 4471902?</p>",
]
for d in demo:
    print("raw    :", d)
    print("clean  :", clean_text(d, placeholder_map, normalise_chat=True))
    print("tokens :", preprocess_query(d, placeholder_map))
    print()
print("chat-speak table: {} entries, e.g.".format(len(CHAT_NORMALISATION)),
      {k: CHAT_NORMALISATION[k] for k in ["u", "pls", "asap", "recieve", "cant"]})
""")

md("""
**c) Removing automated and duplicate messages.** A regex filter drops
auto-replies ("this is an automated message", "do not reply", out-of-office
notices); exact duplicate `(query, reply)` pairs are removed; degenerate rows
(fewer than 2 query tokens or 3 reply tokens) are dropped; and at most **4**
replies are kept per distinct query so that the most frequent intents do not
dominate the loss.
""")

code("""
tests = ["This is an automated message, please do not reply to this email.",
         "Your ticket has been created automatically. Out of office until Monday.",
         "I understand you would like to cancel your order."]
for t in tests:
    print("{!r:70} -> automated: {}".format(t[:66], bool(RE_AUTOMATED.search(t))))
""")

md("""
**d) Sentence-aware truncation.** Half the replies are longer than the 120-token
decoder budget. A hard cut would train the model to stop mid-sentence, so the
reply is cut at the **last complete sentence** that fits.
""")

code("""
lengths = raw["response"].head(200).str.split().str.len()
longest = raw["response"][lengths.idxmax()]
tokens = tokenize(clean_text(longest, placeholder_map))
cut = truncate_at_sentence(tokens, C.MAX_TGT_LEN - 2)
print("original : {} tokens".format(len(tokens)))
print("kept     : {} tokens (budget {})".format(len(cut), C.MAX_TGT_LEN - 2))
print("ends with: ... {}".format(" ".join(cut[-14:])))
print("\\nfraction of the corpus affected: {}/{} replies".format(
    json.load(open(C.STATS_JSON))["truncated_responses"],
    json.load(open(C.STATS_JSON))["pairs_after_cleaning"]))
""")

md("""
**e) Special tokens, padding and the splits.** `<pad> <sos> <eos> <unk>` are
pinned to ids 0–3. Queries are padded/truncated to 32 tokens and replies to 120.
The split is **group-aware**: every copy of a given query lands in the same
split, so no paraphrase leaks from train into test. The vocabulary is built from
the **training split only**, with a minimum frequency of 2.
""")

code("""
stats = json.load(open(C.STATS_JSON))
print(json.dumps(stats, indent=2))
""")

md("""
**f) A separate scope lexicon.** The out-of-scope gate must not use the model
vocabulary — that vocabulary contains every word of every *agent reply*, so
almost any English sentence looks familiar to it. `preprocess.py` therefore
writes a second, much tighter lexicon built from the training **queries** only.
""")

code("""
lexicon = json.load(open(C.SCOPE_LEXICON_JSON))
print("full model vocabulary      : {} types".format(
    len(json.load(open(C.VOCAB_JSON))["itos"])))
print("query-side scope lexicon   : {} types".format(len(lexicon["query_vocab"])))
print("stopwords excluded         : {}".format(len(lexicon["stopwords"])))
print("\\ntop domain anchors:", ", ".join(lexicon["anchors"][:24]))
""")

code("""
from vocab import Vocabulary
vocab = Vocabulary.load(C.VOCAB_JSON)
print("vocabulary size :", len(vocab))
print("special tokens  :", vocab.itos[:4])
print("most frequent   :", vocab.itos[4:24])

train_df = pd.read_csv(C.TRAIN_CSV)
valid_df = pd.read_csv(C.VALID_CSV)
test_df  = pd.read_csv(C.TEST_CSV)
print("\\nsplit sizes -> train {}, valid {}, test {}".format(
    len(train_df), len(valid_df), len(test_df)))

overlap = set(train_df["query_clean"]) & set(test_df["query_clean"])
print("queries shared between train and test:", len(overlap), "(group-aware split)")

ids = vocab.encode(train_df["query_clean"][0].split(), C.MAX_SRC_LEN)
print("\\nencoded example:", train_df["query_clean"][0])
print("ids[:14]:", ids[:14], "... padded to", len(ids))
print("decoded  :", vocab.decode(ids))
""")

code("""
pd.DataFrame({
    "query":  [detokenize(q.split(), placeholder_map) for q in train_df["query_clean"][:5]],
    "reply":  [detokenize(r.split(), placeholder_map)[:180] + " ..." for r in train_df["response_clean"][:5]],
    "intent": train_df["intent"][:5],
})
""")

# --------------------------------------------------------------------------- #
md("""
---
# 3. Model Development

## 3.1 Architecture

Two encoder–decoder models are implemented in `src/model.py`; both expose the
same interface so training, decoding and the app are architecture-agnostic.

**Transformer (default).** 3 encoder + 3 decoder layers, 4 attention heads,
`d_model` 256, feed-forward 512, pre-norm, sinusoidal positional encodings,
input/output embeddings tied. Attention is intrinsic: encoder self-attention,
decoder masked self-attention, and encoder–decoder cross-attention.

**LSTM + Bahdanau attention.** A 1-layer BiLSTM encoder (512 hidden per
direction), a 1-layer LSTM decoder with *input feeding*, and additive attention
`score(hᵈ, hᵉ) = vᵀ tanh(W_d hᵈ + W_e hᵉ)` over the encoder states.
""")

code("""
from model import build_model, count_parameters

for arch in ["transformer", "lstm_attn"]:
    m = build_model(arch, len(vocab), C)
    print("{:<12} {:>10,} trainable parameters".format(arch, count_parameters(m)))

model_preview = build_model("transformer", len(vocab), C)
print()
print(model_preview.transformer.encoder.layers[0])
""")

md("""
## 3.2 Training

AdamW (lr 5e-4, β = 0.9/0.98, weight decay 1e-4), 400 warm-up steps then cosine
decay, label smoothing 0.1, gradient clipping at 1.0, batch size 64, early
stopping on validation loss (patience 4). Perplexity is computed from an
**unsmoothed** cross-entropy, so `exp(loss)` is a true perplexity.

Training was run from the command line:

```bash
python src/train.py --arch transformer --epochs 25
```

The cell below loads the recorded history rather than retraining.
""")

code("""
hist = json.load(open(C.HISTORY_JSON))
print("Training configuration")
for k, v in hist["config"].items():
    print("   {:<18} {}".format(k, v))

history_df = pd.DataFrame(hist["history"])
print()
history_df
""")

code("""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = hist["history"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], "o-", label="train")
ax[0].plot([r["epoch"] for r in h], [r["valid_loss"] for r in h], "s-", label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy per token")
ax[0].set_title("Training / validation loss"); ax[0].grid(alpha=.3); ax[0].legend()
ax[1].plot([r["epoch"] for r in h], [r["valid_ppl"] for r in h], "s-", color="crimson")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("perplexity"); ax[1].set_yscale("log")
ax[1].set_title("Validation perplexity (log scale)"); ax[1].grid(alpha=.3)
fig.tight_layout()
plt.show()

best = min(h, key=lambda r: r["valid_loss"])
print("best epoch {}: valid loss {:.4f}, perplexity {:.2f}".format(
    best["epoch"], best["valid_loss"], best["valid_ppl"]))
""")

md("""
## 3.3 Decoding

`src/decode.py` implements greedy decoding and beam search. Beam search ranks
hypotheses by a **length-normalised** score, `Σ log p / lengthᵃ` with `a = 0.7`
(Wu et al., 2016), so that short generic replies do not always win. A
**no-repeat-3-gram** constraint suppresses the classic seq2seq degeneration loop.
""")

code("""
from decode import ResponseGenerator
gen = ResponseGenerator(checkpoint=C.CHECKPOINT)
print("loaded:", gen.train_config["arch"], "|",
      "{:,} parameters".format(gen.train_config["parameters"]), "| device:", gen.device)
""")

code("""
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
""")

# --------------------------------------------------------------------------- #
md("""
---
# 4. Application Development

`app/app.py` is a Streamlit application launched with:

```bash
streamlit run app/app.py     #  ->  http://localhost:8501
```

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
""")

code("""
import glob
from IPython.display import Image, display, Markdown

shots = sorted(glob.glob(os.path.join(C.REPORT_DIR, "screenshots", "*.png")))
if not shots:
    display(Markdown("> **No screenshots found.** Start the app with "
                     "`streamlit run app/app.py` and save PNGs into "
                     "`reports/screenshots/`."))
for path in shots:
    display(Markdown("**{}**".format(os.path.basename(path))))
    display(Image(filename=path, width=1000))
""")

md("""
## 4.2 The batch path, executed here

The same `ResponseGenerator` the web app uses is called below on the sample file
shipped in `samples/`, which is what the Batch tab does internally.
""")

code("""
with open(os.path.join(C.SAMPLE_DIR, "sample_queries.txt")) as fh:
    batch_queries = [ln.strip() for ln in fh if ln.strip()]

batch_results = gen.generate_batch(batch_queries, strategy="greedy",
                                   apply_scope_check=True)
batch_df = pd.DataFrame([{
    "query": r["query"],
    "generated_response": r["response"][:150] + ("..." if len(r["response"]) > 150 else ""),
    "in_scope": r["in_scope"],
    "oov": r["oov_ratio"],
    "logprob": r["avg_logprob"],
} for r in batch_results])
batch_df
""")

# --------------------------------------------------------------------------- #
md("""
---
# 5. Evaluation and Demonstration

## 5.1 Automatic metrics

Computed by `src/evaluate.py` on the held-out test split:

* **BLEU** and **chrF** (sacreBLEU) — n-gram overlap with the reference reply.
* **ROUGE-1 / ROUGE-2 / ROUGE-L** F1 — recall-oriented overlap, the more
  informative family for long replies.
* **Perplexity** — `exp` of the mean token cross-entropy of the *reference*
  replies under the model; measures the language model itself, independent of the
  decoding strategy.
* **distinct-1 / distinct-2** and the repeated-reply rate — these expose the
  "safe generic answer" failure mode that seq2seq chatbots are notorious for.
""")

code("""
metrics = json.load(open(C.METRICS_JSON))
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
""")

md("""
## 5.2 Manual relevance rating

`src/evaluate.py` writes `reports/manual_rating_sheet.csv`: a random sample of
test queries with the reference and the generated reply, and empty columns for
**relevance (1–5)**, **fluency (1–5)** and **would-send-as-is (yes/no)**. The
filled-in sheet and its averages are reported in `reports/report.md`.
""")

code("""
sheet_path = os.path.join(C.REPORT_DIR, "manual_rating_sheet.csv")
sheet = pd.read_csv(sheet_path)
print("{} rows for manual rating -> {}".format(len(sheet), sheet_path))
sheet[["query", "generated_reply"]].head(5)
""")

md("""
## 5.3 Demonstration, including an out-of-domain query

The last three queries below are deliberately outside the support domain.
""")

code("""
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
""")

code("""
gate = metrics["scope_gate"]
print("thresholds:", gate["thresholds"])
print("in-domain queries accepted   : {:.1%}".format(gate["in_domain_accepted"]))
print("out-of-domain queries refused: {:.1%}".format(gate["out_of_domain_rejected"]))
pd.DataFrame(gate["out_of_domain_detail"])
""")

md("""
### What an out-of-domain query produces **without** the gate

This is the argument for having the gate at all: the decoder never abstains on
its own — it always emits fluent, confident, and wrong support language.
""")

code("""
for q in ["what is the boiling point of water on mars",
          "who won the football world cup in 1998"]:
    r = gen.generate(q, strategy="beam", apply_scope_check=False)
    print("QUERY :", q)
    print("RAW   :", textwrap.fill(r["response"], 96, subsequent_indent="        "))
    print()
""")

# --------------------------------------------------------------------------- #
md("""
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
`decode.py` removes almost all of it; the remaining repetition is inter-sentential
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
seq2seq was trained from scratch (an LSTM + Bahdanau attention variant lives in
the same codebase and is selectable with `--arch lstm_attn`, but every number
reported above is the Transformer's); greedy and length-normalised beam decoding
were implemented with repetition control; and the model was served through a
Streamlit chat application with a
batch upload mode, per-reply diagnostics and an explicit out-of-scope refusal
path. The system drafts fluent, intent-appropriate support replies for the eleven
covered categories and abstains on anything else.

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
""")


# --------------------------------------------------------------------------- #
def main():
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("wrote {} ({} cells)".format(OUT, len(cells)))


if __name__ == "__main__":
    main()
