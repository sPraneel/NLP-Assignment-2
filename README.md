# Customer Support Response Generation Chatbot

An end-to-end conversational **response generation** application. A customer
query goes in, an encoder–decoder network drafts a support reply token by token,
and a Streamlit web app serves it through a chat interface with a batch
(file-upload) mode.

The reply is **generated**, not retrieved: no template table, no nearest-neighbour
lookup. Two interchangeable architectures are implemented — a Transformer
encoder–decoder (default) and an LSTM encoder–decoder with Bahdanau attention.

> **Course:** M.Tech. AIML — Natural Language Processing (S2-25_AIMLCZG530)
> **Assignment 2, GS-3.** Group details are listed at the top of the notebook.

---

## 1. Quick start

Everything lives in **one notebook** and **one data folder**:

```bash
# 1. Python 3.9 or newer (tested on 3.9.6 / 3.10 / 3.11 / 3.12)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Open the notebook and Run All
jupyter lab customer_support_response_generation.ipynb
```

The notebook is **self-contained**: it defines every function it uses inline and
reads and writes nothing but `data/`. It does not import `app.py` or anything in
`scripts/`. Three switches in its setup cell control how much is recomputed:

| Switch | Default | Effect |
|---|---|---|
| `REBUILD_DATA` | `True` | rebuild the splits/vocabulary from the raw corpus (~1 min) |
| `RETRAIN_MODEL` | `False` | reuse `data/best_model.pt`; `True` trains from scratch (~30 min on GPU/MPS) |
| `RECOMPUTE_METRICS` | `False` | reuse `data/metrics.json` if it exists, otherwise evaluate (~5 min) |

A full *Run All* with the defaults takes about three minutes.

### The web application

```bash
streamlit run app.py               # -> http://localhost:8501
```

It loads `data/best_model.pt` and answers exactly as the notebook does.

### Command-line pipeline (optional)

`scripts/` holds standalone copies of the same stages for people who prefer a
terminal. **The notebook never needs them.**

```bash
python scripts/download_dataset.py                        # fetch the raw corpus
python scripts/preprocess.py                              # ~30 s
python scripts/train.py --arch transformer --epochs 25    # ~55 min on an M1
python scripts/evaluate.py --limit 400 --strategy both    # ~5 min
python scripts/run_all.py                                 # all four in order
python scripts/build_notebook.py                          # regenerate the notebook
```

### Running it on the BITS OSHA virtual lab

The project has no OS-specific dependencies; every path is derived from the
project folder, so it runs unchanged in the lab. Two practical notes:

* **Training on CPU.** The device is picked automatically (CUDA → MPS → CPU). On
  a CPU-only machine a 25-epoch run takes a few hours, so leave
  `RETRAIN_MODEL = False` and use the shipped `data/best_model.pt`.
* **Port forwarding.** If the lab exposes a different port, start the app with
  `streamlit run app.py --server.port <PORT> --server.address 0.0.0.0`.

---

## 2. Dataset

| | |
|---|---|
| **Name** | Bitext – Customer Service Tagged Training Dataset for LLM-based Virtual Assistants |
| **Source** | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset |
| **Licence** | Community Data License Agreement – Sharing, v1.0 (CDLA-Sharing-1.0) |
| **Size** | 26,872 query–response pairs · 18 MB · single CSV |
| **Columns** | `flags`, `instruction` (customer message), `category`, `intent`, `response` (agent reply) |
| **Coverage** | 11 categories (ORDER, REFUND, ACCOUNT, PAYMENT, INVOICE, SHIPPING, DELIVERY, SUBSCRIPTION, CANCEL, CONTACT, FEEDBACK) and 27 intents |
| **Message length** | queries 1–16 words (mean 8.7); replies 9–402 words (mean 105) |

This matches the dataset description in the problem statement: a dialogue corpus
of query–response pairs, in the 10 k–1 L range, with optional intent/category
metadata.

After preprocessing: **20,967 train / 2,650 validation / 2,627 test** pairs and a
**3,901-type** shared vocabulary covering 99.87 % of all running tokens. Exact
counts are written to `data/preprocessing_stats.json`.

---

## 3. Repository layout

```
.
├── customer_support_response_generation.ipynb   # THE notebook - self-contained
├── app.py                          # Streamlit web application (Task 4)
├── requirements.txt
├── README.md
├── data/                           # the only folder the notebook touches
│   ├── bitext_customer_support_27k.csv    # raw corpus (18 MB)
│   ├── train.csv / valid.csv / test.csv   # cleaned, leakage-free splits
│   ├── vocab.json                         # word-level vocabulary
│   ├── placeholder_map.json               # {{Order Number}} -> <ph_order_number>
│   ├── scope_lexicon.json                 # query-side lexicon for the scope gate
│   ├── preprocessing_stats.json           # every count reported in Section 2
│   ├── best_model.pt                      # trained Transformer checkpoint
│   ├── history.json                       # per-epoch loss history
│   ├── loss_curve.png                     # training / validation curves
│   ├── metrics.json                       # BLEU, ROUGE, perplexity, scope gate
│   ├── manual_rating_sheet.csv            # sheet for the human relevance rating
│   ├── sample_queries.txt / .csv          # inputs for the batch tab
│   └── screenshots/                       # application screenshots
├── scripts/                        # optional CLI copies - the notebook needs none
│   ├── config.py                   # paths and hyper-parameters
│   ├── vocab.py                    # word-level vocabulary
│   ├── preprocess.py               # cleaning, normalisation, splits (Task 2)
│   ├── data.py                     # Dataset / DataLoader / device selection
│   ├── model.py                    # Transformer and LSTM+attention (Task 3.1)
│   ├── train.py                    # training loop and loss curves (Task 3.2)
│   ├── decode.py                   # greedy + beam search, scope gate (Task 3.3)
│   ├── evaluate.py                 # BLEU / ROUGE / perplexity (Task 5.1)
│   ├── download_dataset.py         # fetch the raw corpus
│   ├── run_all.py                  # download -> preprocess -> train -> evaluate
│   └── build_notebook.py           # regenerate the notebook above
└── reports/                        # previous write-up, kept for reference
    ├── report.md / report.html / Group1.pdf
    └── train_log_transformer.txt
```

`app.py` imports from `scripts/`; the notebook imports from neither. The code is
therefore duplicated once — deliberately, so that the notebook alone is a
complete, runnable submission.

---

## 4. How it works

### Preprocessing (notebook §2.2 · `scripts/preprocess.py`)

1. **Placeholder normalisation.** The corpus marks variable data as
   `{{Order Number}}`, `{{Account Type}}`, … Each of the 43 frequent slots
   becomes one vocabulary token (`<ph_order_number>`); the long tail collapses
   into `<ph_details>`. The map is saved so replies can be rendered back with
   readable `{{Order Number}}` placeholders.
2. **Cleaning.** HTML tags, URLs (`<url>`), e-mail addresses (`<email>`),
   `@usernames` (`<user>`), hashtags, emoji and zero-width characters are
   removed or replaced; curly quotes and dashes are normalised; runs of repeated
   characters (`heyyyy`) and punctuation (`???`) are collapsed; everything is
   lower-cased.
3. **Chat-speak normalisation** (query side only): `u → you`, `pls → please`,
   `asap → as soon as possible`, `recieve → receive`, …
4. **Reference-number slotting**: a live query such as *"where is my order
   4471902"* maps the digit run onto the slot implied by the surrounding words
   (`order` → `<ph_order_number>`, `invoice` → `<ph_invoice_number>`,
   `track` → `<ph_tracking_number>`), so real customer input does not become an
   unknown token. This runs on the **token** stream, not the raw string, because
   the corpus also glues identifiers to the preceding word
   (`cancel purchase370795561790`). Short numbers ("30 days", "24/7") are left
   alone — the threshold is 4 digits.
5. **Automated & duplicate removal.** Regex filters for auto-replies
   ("this is an automated message", "do not reply", out-of-office …), exact
   duplicate pairs, degenerate rows, and a cap of 4 replies per distinct query
   so frequent intents do not dominate the loss.
6. **Tokenisation** with a regex word tokeniser that keeps `<...>` tokens whole.
7. **Sentence-aware truncation.** Replies longer than the decoder budget are cut
   at the last complete sentence that fits, never mid-sentence.
8. **`<sos>` / `<eos>` / `<pad>` / `<unk>`**, fixed-length padding
   (32 source, 120 target tokens), and a **group-aware 80/10/10 split** — all
   copies of a given query land in the same split, so no paraphrase leaks from
   train into test. The vocabulary is built from the **training split only**.
9. A separate **scope lexicon** (`scope_lexicon.json`) is written from the
   training *queries* alone, and drives the out-of-scope gate at inference time.

### Model (notebook §3.1 · `scripts/model.py`)

| | Transformer (default) | LSTM + attention |
|---|---|---|
| Encoder | 3 layers, 4 heads, pre-norm | 1-layer BiLSTM, 512 hidden |
| Decoder | 3 layers, 4 heads | 1-layer LSTM with input feeding |
| Attention | multi-head self + cross | additive (Bahdanau) |
| Embeddings | 256-d, sinusoidal positions, tied with the output layer | 256-d, tied |
| Parameters | ≈ 4.95 M | ≈ 7.9 M |

### Training (notebook §3.2 · `scripts/train.py`)

AdamW (lr 5e-4, β = 0.9/0.98, weight decay 1e-4), linear warm-up over 400 steps
then cosine decay, label smoothing 0.1, gradient clipping at 1.0, batch size 64,
early stopping on validation loss with patience 4. Perplexity is reported from an
**unsmoothed** cross-entropy so `exp(loss)` is meaningful. Curves are written to
`data/loss_curve.png`.

### Decoding and the out-of-scope gate (notebook §3.3 · `scripts/decode.py`)

* **Greedy** decoding (used for batch mode — it batches) and **beam search**
  with length-normalised scoring, `score = Σ log p / lengthᵃ`, `a = 0.7`.
* A **no-repeat-3-gram** constraint, which is what stops the classic seq2seq
  loop ("I'm here to help you. I'm here to help you. …").
* **Out-of-scope detection** uses two cheap, explainable signals:
  1. the fraction of **content words** (non-stopword, > 2 characters) that never
     occur in the **query side of the training split** — > 50 % ⇒ refuse;
  2. the decoder's mean per-token log-probability — < −1.5 ⇒ refuse.

  Signal 1 does the work. Measuring it against the *full* model vocabulary does
  not work, because that vocabulary also contains every word of every agent
  reply and makes almost any English sentence look familiar; the query-side
  lexicon (`data/scope_lexicon.json`, 1,035 words) is far tighter.
  A refused query gets a hand-off message instead of an invented answer.
  Measured separation on the sample queries: in-domain content-OOV ≤ 0.33,
  out-of-domain ≥ 0.60.

---

## 5. Using the web application

**Chat tab** — type a question, press Enter. The transcript stays on screen, and
each reply carries a *Diagnostics* panel showing the decoding strategy, mean
log-probability, unknown-word ratio, the in/out-of-scope decision and the
latency. The transcript can be downloaded as a `.txt`.

**Batch / file upload tab** — upload `data/sample_queries.txt` (one query per
line) or `data/sample_queries.csv` (pick the column holding the queries). The
app answers every row, shows a table with the scope flag and confidence per row,
and offers the results as a downloadable CSV.

**Sidebar** — checkpoint selection, greedy vs beam, beam size, maximum reply
length, an on/off switch for the out-of-scope refusal, and a diagnostics toggle.

---

## 6. Reproducing the reported numbers

Run the notebook with `RETRAIN_MODEL = True` and `RECOMPUTE_METRICS = True`, or
from a terminal:

```bash
python scripts/preprocess.py                               # deterministic (seed 42)
python scripts/train.py --arch transformer --epochs 25     # the reported model
python scripts/evaluate.py --limit 400 --strategy both
```

Every reported number comes from the **Transformer** checkpoint. The LSTM +
Bahdanau attention model is implemented and trains from the same command, but was
not trained for the submission:

```bash
python scripts/train.py --arch lstm_attn --epochs 15
python scripts/evaluate.py --checkpoint data/best_model_lstm_attn.pt --limit 400
```

Everything is seeded with `RANDOM_SEED = 42`. Small differences between machines
are expected because cuDNN/MPS kernels are not bit-deterministic.

Useful flags:

```bash
python scripts/train.py --epochs 2 --limit-batches 20     # 1-minute smoke test
python scripts/decode.py "i want to cancel my order"      # generate from the CLI
python scripts/evaluate.py --strategy greedy --limit 200  # faster evaluation
```

---

## 7. Known issues and limitations

* **Placeholders in the output.** The corpus is written with `{{Order Number}}`
  style slots, so generated replies contain them too. That is faithful to the
  training data and is exactly what an agent-assist draft should look like — a
  human fills the slots before sending. It is *not* a bug.
* **Single-turn.** Each query is answered independently; the transcript is
  displayed but is not fed back to the encoder. Multi-turn context would need
  the corpus to be re-flattened with dialogue history.
* **No grounding.** The model has no access to order, payment or account
  systems. It drafts the *language* of a reply, never a fact. Replies must be
  reviewed before being sent to a customer.
* **Chat-speak normalisation is whitespace-based**, so a slang word glued to
  punctuation (`user info?`) escapes it and is only normalised if the text is
  processed a second time. This affects 3 of the 26,244 pairs and is left as-is
  rather than triggering a full retrain.
* **First launch is slow** (~10 s) while the checkpoint is loaded; Streamlit
  caches it afterwards.
* **Apple silicon.** Install a native (arm64) Python. An x86_64 Homebrew Python
  running under Rosetta has no PyTorch wheels for 3.13+.
* **Beam search on CPU** takes ~1–3 s per reply. Use greedy decoding in the
  sidebar if the lab machine is slow.
* If Streamlit reports *"Port 8501 is already in use"*, run
  `streamlit run app.py --server.port 8502`.

---

## 8. References

1. Sutskever, Vinyals & Le. *Sequence to Sequence Learning with Neural Networks.* NeurIPS 2014.
2. Bahdanau, Cho & Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
3. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
4. Wu et al. *Google's Neural Machine Translation System.* arXiv:1609.08144, 2016 (length-normalised beam search).
5. Li et al. *A Diversity-Promoting Objective Function for Neural Conversation Models.* NAACL 2016 (distinct-n, generic replies).
6. Papineni et al. *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL 2002.
7. Lin. *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop 2004.
8. Bitext. *Customer Service Tagged Training Dataset for LLM-based Virtual Assistants*, 2024. CDLA-Sharing-1.0.
