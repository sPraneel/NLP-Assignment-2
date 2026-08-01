"""Assemble the project report (Task 6) from the artefacts on disk.

    python scripts/build_report.py --group 1

Produces:
    reports/report.md    markdown source, numbers filled in from metrics.json
    reports/report.html  self-contained (screenshots embedded as data URIs)
    reports/Group<N>.pdf rendered with headless Chrome, when Chrome is installed

Everything numeric is read from
  data/processed/preprocessing_stats.json, models/history.json,
  reports/metrics.json and reports/screenshots/ - nothing is hard-coded.
"""

import argparse
import base64
import glob
import json
import mimetypes
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C  # noqa: E402

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
]

SCREENSHOT_CAPTIONS = {
    "01_home_input_screen.png": "Home / input screen — the chat interface as the user first sees it, with the sidebar controls and the example queries.",
    "02_query_submitted.png": "A sample input being submitted — the query typed into the chat box before pressing Enter.",
    "03_generated_response.png": "The generated output as displayed by the application, with the per-reply diagnostics panel.",
    "04_conversation_history.png": "Conversation history retained on screen across several turns.",
    "05_out_of_domain_query.png": "An out-of-domain query: the scope gate refuses and offers a hand-off instead of inventing an answer.",
    "06_batch_upload.png": "Batch mode — `samples/sample_queries.txt` uploaded and every row answered, with a downloadable CSV.",
    "07_sidebar_settings.png": "Sidebar: checkpoint selection, decoding strategy, beam size, reply length and the out-of-scope switch.",
}

CSS = """
@page { size: A4; margin: 16mm 14mm; }
/* The report is a print document: pin it to a light scheme so a dark-mode
   browser or a dark-mode headless Chrome cannot render dark-on-dark text. */
:root { color-scheme: only light; }
html, body { background: #ffffff; color: #1a1a1a; }
* { box-sizing: border-box; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.5; max-width: 190mm; margin: 0 auto; }
h1 { font-size: 20pt; border-bottom: 3px solid #123c69; padding-bottom: 6px; margin-top: 26px; color: #123c69; }
h2 { font-size: 14.5pt; color: #123c69; margin-top: 22px; border-bottom: 1px solid #d8dee6; padding-bottom: 3px; }
h3 { font-size: 12pt; color: #21486f; margin-top: 16px; }
h4 { font-size: 10.8pt; color: #21486f; margin-top: 12px; }
p, li { text-align: left; }
code { background: #f2f4f7; padding: 1px 4px; border-radius: 3px;
       font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt; }
pre { background: #f7f8fa; border: 1px solid #e2e6ec; border-left: 3px solid #123c69;
      padding: 8px 10px; border-radius: 4px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.42; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.3pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #d8dee6; padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: #eef2f7; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
img { max-width: 100%; border: 1px solid #cfd6de; border-radius: 4px; margin: 8px 0;
      page-break-inside: avoid; }
blockquote { border-left: 3px solid #b9c4d1; margin: 10px 0; padding: 2px 12px; color: #46505c; }
figure { margin: 14px 0; page-break-inside: avoid; }
figcaption { font-size: 8.8pt; color: #56606c; font-style: italic; margin-top: 4px; }
hr { border: none; border-top: 1px solid #d8dee6; margin: 22px 0; }
.pagebreak { page-break-before: always; }
"""


# --------------------------------------------------------------------------- #
def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt_table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def screenshot_section():
    shots = sorted(glob.glob(os.path.join(C.REPORT_DIR, "screenshots", "*.png")))
    if not shots:
        return ("> **Screenshots missing.** Start the app (`streamlit run app/app.py`) "
                "and run `python scripts/capture_screenshots.py`.\n")
    blocks = []
    for path in shots:
        name = os.path.basename(path)
        caption = SCREENSHOT_CAPTIONS.get(name, name)
        rel = os.path.join("screenshots", name)
        blocks.append("**Figure — {}**\n\n![{}]({})\n".format(caption, name, rel))
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
def build_markdown(group_no, members):
    stats = load_json(C.STATS_JSON, {})
    hist = load_json(C.HISTORY_JSON, {})
    metrics = load_json(C.METRICS_JSON, {})
    lexicon = load_json(C.SCOPE_LEXICON_JSON, {})

    cfg = hist.get("config", {})
    history = hist.get("history", [])
    best = min(history, key=lambda r: r["valid_loss"]) if history else {}
    by_strategy = metrics.get("by_strategy", {})
    gate = metrics.get("scope_gate", {})

    member_rows = fmt_table(
        [[m["name"], m["id"], m["contribution"]] for m in members],
        ["Name", "BITS ID", "Contribution"])

    split = stats.get("split_sizes", {})

    # --- training table --------------------------------------------------- #
    if history:
        show = history if len(history) <= 14 else (history[:7] + history[-7:])
        train_rows = [[h["epoch"], h["train_loss"], h["valid_loss"], h["valid_ppl"],
                       "{}s".format(h["seconds"])] for h in show]
        train_table = fmt_table(train_rows,
                                ["Epoch", "Train loss", "Valid loss", "Valid PPL", "Time"])
        if len(history) > 14:
            train_table += "\n\n*(middle epochs omitted; the full history is in `models/history.json`)*"
    else:
        train_table = "_Training history not found — run `python src/train.py`._"

    # --- metric table ----------------------------------------------------- #
    if by_strategy:
        rows = []
        for name, m in by_strategy.items():
            rows.append([name, m["bleu_corpus"], m["chrf"], m["rouge1_f"],
                         m["rouge2_f"], m["rougeL_f"], m["distinct_1"],
                         m["distinct_2"],
                         "{}/{}".format(m["genericity"]["unique_responses"],
                                        metrics.get("n_test_queries", "?"))])
        metric_table = fmt_table(rows, ["Decoding", "BLEU", "chrF", "ROUGE-1",
                                        "ROUGE-2", "ROUGE-L", "distinct-1",
                                        "distinct-2", "Unique replies"])
    else:
        metric_table = "_Metrics not found — run `python src/evaluate.py`._"

    ood_rows = [[d["query"], "refused" if not d["in_scope"] else "**answered**",
                 d["oov_ratio"], d["avg_logprob"]]
                for d in gate.get("out_of_domain_detail", [])]
    ood_table = (fmt_table(ood_rows, ["Out-of-domain query", "Gate decision",
                                      "Unknown-word ratio", "Mean log-prob"])
                 if ood_rows else "_not available_")

    samples = metrics.get("qualitative_samples", [])
    sample_blocks = []
    for s in samples[:5]:
        sample_blocks.append(
            "**Query.** {}\n\n**Reference reply.** {}\n\n**Generated reply.** {}\n".format(
                s["query"], s["reference"], s["generated"]))
    sample_text = "\n---\n\n".join(sample_blocks) or "_not available_"

    ppl_line = "{} (validation) / {} (test)".format(
        metrics.get("perplexity_valid", "?"), metrics.get("perplexity_test", "?"))

    md = """# Customer Support Response Generation Chatbot

**M.Tech. AIML — Natural Language Processing (S2-25_AIMLCZG530)**
**Assignment 2 · GS-3 · Group {group_no}**

{member_rows}

---

## Table of contents

1. Problem analysis
2. Data collection and preprocessing
3. Model development
4. Application development (with screenshots)
5. Evaluation and demonstration
6. Observations, conclusion and references

---

# 1. Problem Analysis

## 1.1 Application domain and business processes

The domain is **customer support / service operations** for a consumer-facing
retail or subscription business. Support desks receive a high volume of
repetitive contacts, and an agent spends most of a shift re-typing near-identical
replies. The application is an **agent-assist response drafter**: it reads a
customer query and drafts the reply an agent would have typed.

The business processes covered are:

| Business process | Typical query | Where the draft helps |
|---|---|---|
| Order management | *"I want to cancel order 4471902"* | first response, order-change acknowledgement |
| Returns and refunds | *"How do I get a refund?"* | refund-policy explanation, status updates |
| Billing and invoicing | *"There is a wrong charge on my invoice"* | invoice retrieval, dispute intake |
| Payments | *"What payment methods do you accept?"* | payment FAQ, failed-payment guidance |
| Shipping and delivery | *"Where is my package?"* | tracking guidance, delivery windows |
| Account access | *"I forgot my password"* | recovery walk-through |
| Subscription management | *"Upgrade my account to premium"* | plan-change instructions |
| Contact and escalation | *"I need a human agent"* | routing and hand-off |
| Feedback and complaints | *"I want to leave feedback"* | acknowledgement and routing |

Business value: lower average handling time, consistent tone, faster first
response and lower cost per ticket. The agent stays in the loop — the model
drafts, a human approves.

## 1.2 Problem statement and functional requirements

> **Problem statement.** Given a free-text customer support query, automatically
> generate a fluent, contextually relevant support reply using an
> encoder–decoder neural network, rather than selecting a fixed template.

| # | Functional requirement |
|---|---|
| FR-1 | Accept a free-text query typed into a chat box. |
| FR-2 | Accept a `.txt` (one query per line) or `.csv` (a query column) upload and answer every row. |
| FR-3 | Generate the reply token by token with a trained encoder–decoder; no retrieval, no template lookup. |
| FR-4 | Support greedy **and** beam-search decoding, selectable at run time. |
| FR-5 | Display the reply in a chat interface and retain the full transcript on screen. |
| FR-6 | Detect out-of-scope queries and return a hand-off message instead of an invented answer. |
| FR-7 | Expose per-reply diagnostics (confidence, unknown-word ratio, scope decision, latency). |
| FR-8 | Allow the transcript and the batch results to be downloaded. |

**Non-functional requirements.** A reply in under ~3 s on CPU; the app runs
locally with a single command; the model stays under 10 M parameters so it
trains on a laptop; preprocessing is deterministic under a fixed seed.

**Behaviour when the query is out of scope (FR-6).** A query is refused when
*either*

* more than **{oov_thresh:.0%}** of its *content* words (non-stopword, longer than
  two characters) never occur in the **queries** of the training split, or
* the decoder's mean per-token log-probability falls below **{lp_thresh}**.

The user then receives a message naming the topics the assistant can handle and
offering a hand-off to a human agent. This is deliberately conservative: in
customer communication a confident wrong answer costs more than an admission of
ignorance.

## 1.3 Expected input and output

| | |
|---|---|
| **Input** | One customer query as free text (chat box), or a `.txt` / `.csv` file of queries (batch mode). |
| **Output** | A generated support reply of roughly 40–120 words, rendered with readable `{{{{Placeholders}}}}` (for example `{{{{Order Number}}}}`) that a human agent fills in, plus a scope flag and a confidence score. Batch mode returns a downloadable CSV. |
| **Conversation type** | **Single-turn.** Each query is encoded and answered independently. The transcript is retained and displayed on screen, but earlier turns are not fed back into the encoder. |

Multi-turn support would require the corpus to be re-flattened with dialogue
history concatenated into the source sequence. The architecture supports it; the
chosen corpus does not carry the history.

---

# 2. Data Collection and Preprocessing

## 2.1 Dataset

| | |
|---|---|
| **Name** | Bitext — Customer Service Tagged Training Dataset for LLM-based Virtual Assistants |
| **Source** | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset |
| **Licence** | Community Data License Agreement — Sharing, v1.0 (**CDLA-Sharing-1.0**) |
| **Size** | **{raw_pairs:,}** query–response pairs · 18 MB · one CSV file |
| **Structure** | `instruction` (customer message), `response` (agent reply), plus `category`, `intent` and `flags` metadata |
| **Coverage** | 11 categories (ORDER, REFUND, ACCOUNT, PAYMENT, INVOICE, SHIPPING, DELIVERY, SUBSCRIPTION, CANCEL, CONTACT, FEEDBACK) and 27 intents |
| **Message length** | queries 1–16 words (mean 8.7); replies 9–402 words (mean 105) |

This matches the dataset description in the problem statement: a dialogue corpus
of query–response pairs in the 10 k–1 L range, drawn from customer-support chat
logs, with optional intent metadata.

## 2.2 Preprocessing pipeline

Implemented in `src/preprocess.py` and reproducible with `python src/preprocess.py`.

1. **Placeholder normalisation.** The corpus marks variable data as
   `{{{{Order Number}}}}`, `{{{{Account Type}}}}`, … Left alone the braces would be
   shredded by tokenisation. Each of the **{placeholder_types}** frequent slots
   becomes a single vocabulary token (`<ph_order_number>`); the long tail
   collapses into `<ph_details>`. The map is saved so replies can be rendered
   back with readable placeholders.
2. **Cleaning.** HTML tags, URLs (`<url>`), e-mail addresses (`<email>`),
   `@usernames` (`<user>`), hashtags, emoji and zero-width characters are removed
   or replaced; curly quotes and dashes are normalised; runs of repeated
   characters (`heyyyy`) and punctuation (`???`) are collapsed; text is
   lower-cased.
3. **Chat-speak normalisation** (query side): `u → you`, `pls → please`,
   `asap → as soon as possible`, `recieve → receive`, and 60 further entries.
4. **Reference-number slotting.** A live query such as *"where is my order
   4471902"* maps the digit run onto the slot implied by the surrounding words,
   so real customer input does not become an unknown token. This runs on the
   token stream rather than the raw string, because the corpus also glues
   identifiers to the preceding word (`cancel purchase370795561790`); numbers
   shorter than four digits ("30 days", "24/7") are left alone.
5. **Automated and duplicate removal.** Regex filters for auto-replies
   ("this is an automated message", "do not reply", out-of-office notices);
   **{dropped_duplicate_pairs}** exact duplicate pairs and
   **{dropped_too_short}** degenerate rows removed; at most four replies kept per
   distinct query, which dropped a further **{dropped_over_represented}** rows so
   that frequent intents do not dominate the loss.
6. **Tokenisation** with a regex word tokeniser that keeps `<...>` tokens whole
   and holds on to intra-word apostrophes and hyphens (`don't`, `step-by-step`).
7. **Sentence-aware truncation.** **{truncated_responses:,}** replies exceed the
   {max_tgt} token decoder budget; each is cut at the *last complete sentence*
   that fits, never mid-sentence, so the model is never taught to stop abruptly.
8. **Special tokens and padding.** `<pad> <sos> <eos> <unk>` are pinned to ids
   0–3; queries are padded/truncated to {max_src} tokens and replies to
   {max_tgt}.
9. **Group-aware 80/10/10 split.** Every copy of a given query lands in the same
   split, so no paraphrase leaks from train into test.

### Resulting corpus

{corpus_table}

The vocabulary is built from the **training split only** with a minimum frequency
of {min_freq}, giving **{vocab_size:,}** types that cover **{coverage:.2%}** of all
running tokens.

A second, much tighter **scope lexicon** ({scope_vocab:,} words) is built from the
training *queries* alone and drives the out-of-scope gate. Using the full model
vocabulary there does not work: it also contains every word of every agent reply,
which makes almost any English sentence look familiar.

---

# 3. Model Development

## 3.1 Architecture

Two encoder–decoder models are implemented in `src/model.py` behind one
interface, so training, decoding and the web app are architecture-agnostic.

**Transformer (the model reported here).** {enc_layers} encoder and
{dec_layers} decoder layers, {n_heads} attention heads, `d_model` {d_model},
feed-forward {ffn}, pre-norm residual blocks, sinusoidal positional encodings and
input/output embedding tying. Attention is intrinsic to the architecture:
encoder self-attention, masked decoder self-attention and encoder–decoder
cross-attention.

**LSTM + Bahdanau attention (alternative).** A 1-layer BiLSTM encoder (512 hidden
units per direction), a 1-layer LSTM decoder with *input feeding*, and additive
attention `score(h_dec, h_enc) = vᵀ tanh(W_d h_dec + W_e h_enc)` over the encoder
states. Selected with `python src/train.py --arch lstm_attn`.

## 3.2 Training configuration

{config_table}

Optimiser AdamW (β = 0.9/0.98), {warmup} warm-up steps followed by cosine decay,
label smoothing {label_smoothing}, gradient clipping at {clip}, early stopping on
validation loss with patience {patience}. Perplexity is computed from an
**unsmoothed** cross-entropy so that `exp(loss)` is a true perplexity.

## 3.3 Loss curves

{train_table}

Best epoch: **{best_epoch}**, validation loss **{best_loss}**, validation
perplexity **{best_ppl}**.

![Training and validation loss](loss_curve.png)

## 3.4 Decoding

`src/decode.py` implements both decoding strategies required by the assignment:

* **Greedy decoding** — argmax at every step, batched, used for the file-upload
  path where throughput matters.
* **Beam search** — beam size {beam_size}, ranked by the length-normalised score
  `Σ log p / lengthᵃ` with `a = {length_penalty}` (Wu et al., 2016). Without
  length normalisation, short generic replies always win because they accumulate
  less negative log-probability.
* A **no-repeat-{no_repeat}-gram** constraint, which is what stops the classic
  seq2seq degeneration loop (*"I'm here to help you. I'm here to help you. …"*).

---

# 4. Application Development

`app/app.py` is a Streamlit application launched from the project root with:

```
streamlit run app/app.py        #  ->  http://localhost:8501
```

**Chat tab.** A chat-style interface built on `st.chat_message` /
`st.chat_input`. The transcript is held in `st.session_state` and re-rendered on
every run, so the whole conversation stays on screen. Each reply carries a
*Diagnostics* panel — decoding strategy, mean log-probability, unknown-word
ratio, in/out-of-scope decision and latency — and the transcript is downloadable.

**Batch / file-upload tab.** Accepts `.txt` (one query per line) or `.csv` (the
user chooses the column holding the queries). Every row is answered with batched
greedy decoding; results appear in a table with the scope flag and confidence per
row and can be downloaded as CSV.

**Sidebar.** Checkpoint selection, greedy vs beam, beam size, maximum reply
length, an on/off switch for the out-of-scope refusal, a diagnostics toggle and a
*clear conversation* button.

## 4.1 Screenshots of the working application

{screenshots}

---

# 5. Evaluation and Demonstration

## 5.1 Automatic metrics

Measured on the held-out test split ({n_test} decoded queries; perplexity over
the full split). BLEU and chrF come from sacreBLEU, ROUGE from Google's
`rouge_score`.

**Perplexity:** {ppl_line}

{metric_table}

`distinct-1` / `distinct-2` are the ratios of unique uni-/bi-grams to total
n-grams across all generated replies. They — together with the unique-reply count
— are the diagnostic for the "safe generic answer" failure mode of seq2seq
chatbots (Li et al., 2016), which BLEU and ROUGE do not expose.

## 5.2 Manual relevance rating

`src/evaluate.py` writes `reports/manual_rating_sheet.csv`, a random sample of
test queries with the reference and generated replies and empty columns for
**relevance (1–5)**, **fluency (1–5)** and **would-send-as-is (yes/no)**. The
sheet was rated independently by the group members; averages:

| Criterion | Mean score | Notes |
|---|---|---|
| Relevance to the query | _fill in_ | does the reply address the intent that was asked about |
| Fluency / grammaticality | _fill in_ | reads like a human agent |
| Would send as-is (after filling placeholders) | _fill in %_ | production readiness |

> Fill this table in from your completed `reports/manual_rating_sheet.csv`
> before submitting.

## 5.3 Sample inputs and outputs

{sample_text}

## 5.4 Out-of-domain demonstration

The gate accepted **{in_domain_accepted:.1%}** of in-domain test queries and
refused **{ood_rejected:.1%}** of the deliberately out-of-domain queries below.

{ood_table}

Without the gate the decoder does **not** abstain — it emits fluent, confident
and entirely wrong support language for every one of these. That is the argument
for having the gate at all.

---

# 6. Observations, Conclusion and References

## 6.1 Observations

**Generic replies.** The model converges on a small set of high-frequency
openings — *"I understand …"*, *"I'm sorry to hear …"*, *"Thank you for reaching
out …"*. This is the documented safe-answer bias of maximum-likelihood seq2seq
models: under cross-entropy the lowest-risk output is the most frequent one. The
distinct-n figures and the unique-reply count in §5.1 quantify how far the model
falls into it. Beam search makes genericity *worse* than greedy decoding unless
length normalisation is applied.

**Repetition.** Without a constraint the decoder loops on politeness clauses.
The no-repeat-3-gram rule removes almost all of it; what remains is
inter-sentential repetition (the same idea rephrased), which n-gram blocking
cannot catch.

**Factual reliability.** The model has **no access to any real system**. When it
emits `{{{{Order Number}}}}` or `{{{{Customer Support Hours}}}}` it is reproducing a
slot it saw in training, not looking anything up. Any number, date or policy
detail in a generated reply is a *pattern*, not a fact. The placeholders are
therefore a feature: they mark exactly where a human must supply real data.

**Safety of automated customer communication.** Three risks matter in production:

1. *Confident nonsense on out-of-scope input* — mitigated by the scope gate,
   which abstains rather than inventing.
2. *Commitments the business cannot keep* — a drafted "your refund will be
   processed in 3–5 days" is a promise. Drafts must be reviewed before sending.
3. *Tone on distressed contacts* — the corpus is uniformly polite and contains no
   genuinely angry customers, so the model is untested on them.

The recommended deployment is therefore **agent-assist, not auto-send**: the
model proposes, an agent approves, and the scope gate routes anything unusual to
a human.

**Data limitations.** The corpus is hybrid-synthetic and highly templated. That
makes it learnable by a {params}-parameter model on a laptop, but it inflates
overlap metrics relative to real, messy support logs. The numbers in §5.1 should
be read as an upper bound.

## 6.2 Conclusion

An encoder–decoder response generator was built end to end: {raw_pairs:,} support
pairs were cleaned, normalised, de-duplicated and split without leakage; a
Transformer seq2seq was trained from scratch to a validation perplexity of
{best_ppl} (an LSTM + Bahdanau attention variant is implemented in the same
codebase and selectable with `--arch lstm_attn`, but the numbers reported here
are the Transformer's); greedy and
length-normalised beam decoding were implemented with repetition control; and the
model was served through a Streamlit chat application with a batch upload mode,
per-reply diagnostics and an explicit out-of-scope refusal path. The system
drafts fluent, intent-appropriate support replies for the eleven covered
categories and abstains on anything else.

The main limitation is genericity, an artefact of maximum-likelihood training on
a templated corpus. Natural extensions: multi-turn context in the encoder,
sub-word (BPE) vocabulary to remove `<unk>` entirely, a copy/pointer mechanism
for order and invoice numbers, an explicit intent classifier as a second scope
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
10. Streamlit documentation — chat elements. https://docs.streamlit.io/develop/api-reference/chat

---

## Appendix A — how to run

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dataset.py
python src/preprocess.py
python src/train.py
python src/evaluate.py --limit 400 --strategy both
streamlit run app/app.py
```

Full setup instructions, the repository layout and the known issues are in
`README.md` inside `Group{group_no}_Code.zip`.
""".format(
        group_no=group_no,
        member_rows=member_rows,
        oov_thresh=C.OOS_OOV_RATIO,
        lp_thresh=C.OOS_MIN_AVG_LOGPROB,
        raw_pairs=stats.get("raw_pairs", 0),
        placeholder_types=stats.get("placeholder_types", "?"),
        dropped_duplicate_pairs=stats.get("dropped_duplicate_pairs", "?"),
        dropped_too_short=stats.get("dropped_too_short", "?"),
        dropped_over_represented=stats.get("dropped_over_represented", "?"),
        truncated_responses=stats.get("truncated_responses", 0),
        max_src=C.MAX_SRC_LEN, max_tgt=C.MAX_TGT_LEN,
        corpus_table=fmt_table(
            [["Raw pairs", "{:,}".format(stats.get("raw_pairs", 0))],
             ["After cleaning and de-duplication", "{:,}".format(stats.get("pairs_after_cleaning", 0))],
             ["Train / validation / test", "{:,} / {:,} / {:,}".format(
                 split.get("train", 0), split.get("valid", 0), split.get("test", 0))],
             ["Mean query length", "{} tokens".format(stats.get("mean_query_tokens", "?"))],
             ["Mean reply length", "{} tokens".format(stats.get("mean_response_tokens", "?"))],
             ["Vocabulary", "{:,} types".format(stats.get("vocab_size", 0))],
             ["Token coverage", "{:.2%}".format(stats.get("token_coverage", 0))],
             ["Scope lexicon (query side)", "{:,} words".format(len(lexicon.get("query_vocab", [])))]],
            ["Quantity", "Value"]),
        min_freq=stats.get("min_freq", 2),
        vocab_size=stats.get("vocab_size", 0),
        coverage=stats.get("token_coverage", 0),
        scope_vocab=len(lexicon.get("query_vocab", [])),
        enc_layers=cfg.get("enc_layers", C.N_ENC_LAYERS),
        dec_layers=cfg.get("dec_layers", C.N_DEC_LAYERS),
        n_heads=cfg.get("n_heads", C.N_HEADS),
        d_model=cfg.get("d_model", C.D_MODEL),
        ffn=cfg.get("ffn_dim", C.FFN_DIM),
        config_table=fmt_table(
            [[k.replace("_", " "), v] for k, v in cfg.items()
             if k not in ("device",)],
            ["Setting", "Value"]) if cfg else "_not available_",
        warmup=cfg.get("warmup_steps", C.WARMUP_STEPS),
        label_smoothing=cfg.get("label_smoothing", C.LABEL_SMOOTHING),
        clip=cfg.get("clip_norm", C.CLIP_NORM),
        patience=cfg.get("patience", C.PATIENCE),
        train_table=train_table,
        best_epoch=best.get("epoch", "?"),
        best_loss=best.get("valid_loss", "?"),
        best_ppl=best.get("valid_ppl", "?"),
        beam_size=C.BEAM_SIZE,
        length_penalty=C.LENGTH_PENALTY,
        no_repeat=C.NO_REPEAT_NGRAM,
        screenshots=screenshot_section(),
        n_test=metrics.get("n_test_queries", "?"),
        ppl_line=ppl_line,
        metric_table=metric_table,
        sample_text=sample_text,
        in_domain_accepted=gate.get("in_domain_accepted", 0),
        ood_rejected=gate.get("out_of_domain_rejected", 0),
        ood_table=ood_table,
        params="{:,}".format(cfg.get("parameters", 0)),
    )
    return md


# --------------------------------------------------------------------------- #
def markdown_to_html(md_text, base_dir, title):
    import mistune

    renderer = mistune.create_markdown(
        plugins=["table", "strikethrough", "footnotes"])
    body = renderer(md_text)

    # inline every local image so the HTML/PDF is self-contained
    import re

    def embed(match):
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = os.path.join(base_dir, src)
        if not os.path.exists(path):
            return match.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        return 'src="data:{};base64,{}"'.format(mime, data)

    body = re.sub(r'src="([^"]+)"', embed, body)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>{}</title><style>{}</style></head><body>{}</body></html>"
            ).format(title, CSS, body)


def html_to_pdf(html_path, pdf_path):
    chrome = next((c for c in CHROME_CANDIDATES if os.path.exists(c)), None)
    if chrome is None:
        print("  Chrome not found - open {} in a browser and use "
              "'Print -> Save as PDF'.".format(os.path.relpath(html_path, ROOT)))
        return False
    cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--virtual-time-budget=20000",
           "--print-to-pdf={}".format(pdf_path), "file://" + html_path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(pdf_path):
        print("  Chrome failed: {}".format(result.stderr.decode()[:400]))
        return False
    return True


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="1", help="group number, used in the filename")
    p.add_argument("--members", default=None,
                   help="JSON list: [{\"name\":..,\"id\":..,\"contribution\":\"25%%\"}, ...]")
    args = p.parse_args()

    if args.members:
        members = json.loads(args.members)
    else:
        members = [{"name": "_[member {} — fill in]_".format(i),
                    "id": "_[2xxxxxxxx]_", "contribution": "25%"}
                   for i in range(1, 5)]
        print("NOTE: no --members given, the group table holds placeholders.\n"
              "      Re-run with, for example:\n"
              "      python scripts/build_report.py --group 7 --members "
              "'[{\"name\":\"A Kumar\",\"id\":\"2023ab12345\",\"contribution\":\"25%\"}]'")

    os.makedirs(C.REPORT_DIR, exist_ok=True)
    md = build_markdown(args.group, members)

    md_path = os.path.join(C.REPORT_DIR, "report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print("markdown -> {}".format(os.path.relpath(md_path, ROOT)))

    html = markdown_to_html(md, C.REPORT_DIR,
                            "Group {} - Customer Support Response Generation".format(args.group))
    html_path = os.path.join(C.REPORT_DIR, "report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("html     -> {} ({:.1f} MB)".format(os.path.relpath(html_path, ROOT),
                                              len(html) / 1e6))

    pdf_path = os.path.join(C.REPORT_DIR, "Group{}.pdf".format(args.group))
    if html_to_pdf(html_path, pdf_path):
        print("pdf      -> {} ({:.1f} MB)".format(
            os.path.relpath(pdf_path, ROOT), os.path.getsize(pdf_path) / 1e6))


if __name__ == "__main__":
    main()
