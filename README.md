# Customer Support Response Generation Chatbot

An end-to-end conversational **response generation** application. A customer
query goes in, an encoder–decoder network drafts a support reply token by token,
and a Streamlit web app serves it through a chat interface with a batch
(file-upload) mode.

The reply is **generated**, not retrieved: no template table, no nearest-neighbour
lookup. Two interchangeable architectures are implemented — a Transformer
encoder–decoder (default) and an LSTM encoder–decoder with Bahdanau attention.

> **Course:** M.Tech. AIML — Natural Language Processing (S2-25_AIMLCZG530)
> **Assignment 2, GS-3.** Group details are listed in `reports/report.md`.

---

## 1. Quick start

```bash
# 1. Python 3.9 or newer (tested on 3.9.6 / 3.10 / 3.11 / 3.12)
python3 --version

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Dataset  (~18 MB, downloaded into data/raw/)
python scripts/download_dataset.py

# 5. Preprocess  (~30 s)
python src/preprocess.py

# 6. Train  (~55 min on an Apple M1 / ~15 min on a CUDA GPU)
python src/train.py

# 7. Evaluate  (~5 min)
python src/evaluate.py --limit 400 --strategy both

# 8. Smoke test  (~15 s) — verifies artefacts, decoding and the scope gate
python scripts/selftest.py

# 9. Launch the web application
streamlit run app/app.py
```

The app opens at **http://localhost:8501**.

If `models/best_model.pt` is shipped inside the ZIP you can skip steps 6 and
go straight to step 8 — but you still need steps 4 and 5, because the app loads
`data/processed/vocab.json` and `data/processed/placeholder_map.json`.

### One-command rebuild

```bash
python scripts/run_all.py
```

Runs download → preprocess → train → evaluate in sequence and then prints the
command to start the app.

### Running it on the BITS OSHA virtual lab

The project has no OS-specific dependencies; every path in `src/config.py` is
derived from the repository root, so it runs unchanged in the lab. Two practical
notes:

* **Training on CPU.** `src/data.py` picks CUDA → MPS → CPU automatically. On a
  CPU-only lab machine a full 25-epoch run takes a few hours, so either train
  with fewer epochs (`python src/train.py --epochs 8`, which already reaches a
  usable perplexity) or use the `models/best_model.pt` shipped in the ZIP and go
  straight to `streamlit run app/app.py`.
* **Port forwarding.** If the lab exposes a different port, start the app with
  `streamlit run app/app.py --server.port <PORT> --server.address 0.0.0.0`.

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

After preprocessing: **20,999 train / 2,642 validation / 2,626 test** pairs and a
**3,934-type** shared vocabulary covering 99.88 % of all running tokens. Exact
counts are written to `data/processed/preprocessing_stats.json`.

---

## 3. Repository layout

```
.
├── app/
│   └── app.py                     # Streamlit web application (Task 4)
├── data/
│   ├── raw/                       # downloaded corpus (git-ignored)
│   └── processed/                 # splits, vocab.json, placeholder_map.json
├── models/
│   ├── best_model.pt              # trained Transformer checkpoint
│   └── history.json               # per-epoch loss history
├── notebooks/
│   └── customer_support_response_generation.ipynb   # end-to-end walkthrough
├── reports/
│   ├── loss_curve.png             # training / validation curves
│   ├── metrics.json               # BLEU, ROUGE, perplexity, scope gate
│   ├── manual_rating_sheet.csv    # sheet for the human relevance rating
│   ├── screenshots/               # application screenshots
│   └── report.md                  # project report (Task 6)
├── samples/
│   ├── sample_queries.txt         # 17 queries, one per line
│   └── sample_queries.csv         # same queries with ticket metadata
├── scripts/
│   ├── download_dataset.py        # fetch the raw corpus
│   ├── run_all.py                 # download -> preprocess -> train -> evaluate
│   ├── selftest.py                # 15-second end-to-end smoke test
│   ├── capture_screenshots.py     # drive the running app, save PNGs
│   ├── build_notebook.py          # regenerate the submission notebook
│   ├── build_report.py            # report.md -> report.html -> Group<N>.pdf
│   ├── make_submission_zip.py     # build Group<N>_Code.zip
│   └── build_submission.py        # all of the above, in order
├── src/
│   ├── config.py                  # every path and hyper-parameter
│   ├── vocab.py                   # word-level vocabulary
│   ├── preprocess.py              # cleaning, normalisation, splits (Task 2)
│   ├── data.py                    # Dataset / DataLoader / device selection
│   ├── model.py                   # Transformer and LSTM+attention (Task 3.1)
│   ├── train.py                   # training loop and loss curves (Task 3.2)
│   ├── decode.py                  # greedy + beam search, scope gate (Task 3.3)
│   └── evaluate.py                # BLEU / ROUGE / perplexity (Task 5.1)
├── requirements.txt
└── README.md
```

---

## 4. How it works

### Preprocessing (`src/preprocess.py`)

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

### Model (`src/model.py`)

| | Transformer (default) | LSTM + attention |
|---|---|---|
| Encoder | 3 layers, 4 heads, pre-norm | 1-layer BiLSTM, 512 hidden |
| Decoder | 3 layers, 4 heads | 1-layer LSTM with input feeding |
| Attention | multi-head self + cross | additive (Bahdanau) |
| Embeddings | 256-d, sinusoidal positions, tied with the output layer | 256-d, tied |
| Parameters | ≈ 4.95 M | ≈ 7.9 M |

### Training (`src/train.py`)

AdamW (lr 5e-4, β = 0.9/0.98, weight decay 1e-4), linear warm-up over 400 steps
then cosine decay, label smoothing 0.1, gradient clipping at 1.0, batch size 64,
early stopping on validation loss with patience 4. Perplexity is reported from an
**unsmoothed** cross-entropy so `exp(loss)` is meaningful. Curves are written to
`reports/loss_curve.png`.

### Decoding and the out-of-scope gate (`src/decode.py`)

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
  lexicon (`data/processed/scope_lexicon.json`, 1,050 words) is far tighter.
  A refused query gets a hand-off message instead of an invented answer.
  Measured separation on the sample queries: in-domain content-OOV ≤ 0.33,
  out-of-domain ≥ 0.60.

---

## 5. Using the web application

**Chat tab** — type a question, press Enter. The transcript stays on screen, and
each reply carries a *Diagnostics* panel showing the decoding strategy, mean
log-probability, unknown-word ratio, the in/out-of-scope decision and the
latency. The transcript can be downloaded as a `.txt`.

**Batch / file upload tab** — upload `samples/sample_queries.txt` (one query per
line) or `samples/sample_queries.csv` (pick the column holding the queries). The
app answers every row, shows a table with the scope flag and confidence per row,
and offers the results as a downloadable CSV.

**Sidebar** — checkpoint selection, greedy vs beam, beam size, maximum reply
length, an on/off switch for the out-of-scope refusal, and a diagnostics toggle.

---

## 6. Reproducing the reported numbers

```bash
python src/preprocess.py                                   # deterministic (seed 42)
python src/train.py --arch transformer --epochs 25         # the reported model
python src/evaluate.py --limit 400 --strategy both
```

Every number in `reports/report.md` comes from the **Transformer** checkpoint.
The LSTM + Bahdanau attention model is implemented and trains from the same
command, but was not trained for the submission:

```bash
python src/train.py --arch lstm_attn --epochs 15
python src/evaluate.py --checkpoint models/best_model_lstm_attn.pt --limit 400
```

Everything is seeded with `RANDOM_SEED = 42` in `src/config.py`. Small
differences between machines are expected because cuDNN/MPS kernels are not
bit-deterministic.

Useful flags:

```bash
python src/train.py --epochs 2 --limit-batches 20     # 1-minute smoke test
python src/decode.py "i want to cancel my order"      # generate from the CLI
python src/evaluate.py --strategy greedy --limit 200  # faster evaluation
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
  `streamlit run app/app.py --server.port 8502`.

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
