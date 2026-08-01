"""Task 4 - Streamlit web application for the support response generator.

Launch from the project root with:

    streamlit run app/app.py

The app has two modes:
  * Chat        - single-turn chat interface that keeps the transcript on screen.
  * Batch       - upload a .txt (one query per line) or .csv (a column of
                  queries) and get a drafted reply for every row.
"""

import io
import os
import sys
import time

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C                       # noqa: E402
from decode import ResponseGenerator     # noqa: E402

st.set_page_config(page_title="Customer Support Response Generator",
                   page_icon="💬", layout="wide")

WELCOME = ("Hello! I'm the automated support assistant. Ask me about orders, "
           "refunds, payments, invoices, shipping and delivery, subscriptions, "
           "account access, or contacting a human agent.")

EXAMPLES = [
    "I want to cancel order 4471902",
    "How do I get a refund for my last purchase?",
    "I forgot my password and cannot sign in",
    "Where is my package? It has not arrived yet",
    "I need to talk to a human agent",
    "What is the boiling point of water on Mars?",   # deliberately out of scope
]


# --------------------------------------------------------------------------- #
# Model loading (cached so the checkpoint is read once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the trained encoder-decoder model ...")
def load_generator(checkpoint: str):
    return ResponseGenerator(checkpoint=checkpoint)


def checkpoint_options():
    if not os.path.isdir(C.MODEL_DIR):
        return []
    return sorted(os.path.join(C.MODEL_DIR, f)
                  for f in os.listdir(C.MODEL_DIR) if f.endswith(".pt"))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar():
    st.sidebar.title("⚙️ Settings")

    ckpts = checkpoint_options()
    if not ckpts:
        st.sidebar.error("No checkpoint found in `models/`.\n\n"
                         "Run `python src/train.py` first.")
        st.stop()
    default = C.CHECKPOINT if C.CHECKPOINT in ckpts else ckpts[0]
    checkpoint = st.sidebar.selectbox(
        "Model checkpoint", ckpts, index=ckpts.index(default),
        format_func=os.path.basename)

    st.sidebar.subheader("Decoding")
    strategy = st.sidebar.radio("Strategy", ["beam", "greedy"], index=0,
                                horizontal=True,
                                help="Beam search explores several drafts and "
                                     "usually reads better; greedy is faster.")
    beam_size = st.sidebar.slider("Beam size", 1, 6, C.BEAM_SIZE,
                                  disabled=(strategy != "beam"))
    max_len = st.sidebar.slider("Max reply length (tokens)", 20,
                                C.MAX_DECODE_LEN, C.MAX_DECODE_LEN, step=10)

    st.sidebar.subheader("Safety")
    scope_check = st.sidebar.checkbox(
        "Refuse out-of-scope questions", value=True,
        help="Falls back to a hand-off message when the query is unlike "
             "anything in the support corpus or the decoder is unsure.")
    show_diag = st.sidebar.checkbox("Show model diagnostics", value=True)

    if st.sidebar.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    return {"checkpoint": checkpoint, "strategy": strategy, "beam_size": beam_size,
            "max_len": max_len, "scope_check": scope_check, "show_diag": show_diag}


def model_card(gen):
    cfg = gen.train_config
    st.sidebar.subheader("Model")
    st.sidebar.markdown(
        "- **Architecture:** `{}`\n"
        "- **Parameters:** {:,}\n"
        "- **Vocabulary:** {:,} types\n"
        "- **Trained on:** {:,} query/reply pairs\n"
        "- **Device:** `{}`".format(
            cfg.get("arch", "?"), cfg.get("parameters", 0),
            cfg.get("vocab_size", len(gen.vocab)), cfg.get("train_pairs", 0),
            gen.device))


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
def render_diagnostics(meta):
    cols = st.columns(4)
    cols[0].metric("Decoding", meta.get("strategy", "-"))
    cols[1].metric("Mean log-prob", "{:.2f}".format(meta.get("avg_logprob", 0.0)))
    cols[2].metric("Unknown words", "{:.0%}".format(meta.get("oov_ratio", 0.0)))
    cols[3].metric("In scope", "Yes" if meta.get("in_scope") else "No")
    st.caption("Scope decision: {} · latency {:.2f}s".format(
        meta.get("reason", "-"), meta.get("latency", 0.0)))


def chat_tab(gen, opts):
    if "history" not in st.session_state:
        st.session_state.history = []

    st.caption("Single-turn assistant: every query is answered independently, "
               "and the full transcript stays on screen.")

    with st.expander("💡 Example queries", expanded=not st.session_state.history):
        cols = st.columns(3)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 3].button(example, key="ex{}".format(i),
                                  use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    with st.chat_message("assistant"):
        st.write(WELCOME)

    last = len(st.session_state.history) - 1
    for i, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["response"])
            if opts["show_diag"]:
                with st.expander("Diagnostics", expanded=(i == last)):
                    render_diagnostics(turn)

    query = st.chat_input("Type your question and press Enter ...")
    if not query and st.session_state.get("pending"):
        query = st.session_state.pop("pending")

    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Drafting a reply ..."):
                started = time.time()
                result = gen.generate(query, strategy=opts["strategy"],
                                      beam_size=opts["beam_size"],
                                      max_len=opts["max_len"],
                                      apply_scope_check=opts["scope_check"])
                result["latency"] = time.time() - started
            st.write(result["response"])
        st.session_state.history.append(result)
        # Redraw so the transcript stays above the input box: during the run
        # that generates a reply, Streamlit appends the new turn *after* the
        # already-rendered chat input.
        st.rerun()

    if st.session_state.history:
        transcript = "\n\n".join(
            "Customer: {}\nAssistant: {}".format(t["query"], t["response"])
            for t in st.session_state.history)
        st.download_button("⬇️ Download transcript", transcript,
                           file_name="support_transcript.txt", mime="text/plain")


# --------------------------------------------------------------------------- #
# Batch tab
# --------------------------------------------------------------------------- #
def read_queries(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        if df.empty:
            return pd.DataFrame(columns=["query"])
        text_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
        default = next((c for c in text_cols
                        if c.lower() in ("query", "question", "instruction",
                                         "message", "text", "customer_query")),
                       text_cols[0])
        column = st.selectbox("Which column holds the customer queries?",
                              text_cols, index=text_cols.index(default))
        return pd.DataFrame({"query": df[column].astype(str)})
    raw = uploaded.read().decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return pd.DataFrame({"query": lines})


def batch_tab(gen, opts):
    st.caption("Upload a **.txt** file with one query per line, or a **.csv** "
               "with a column of queries. Every row is answered by the same model.")

    uploaded = st.file_uploader("Query file", type=["txt", "csv"])
    sample_dir = os.path.join(C.ROOT_DIR, "samples")
    if os.path.isdir(sample_dir):
        st.caption("Sample files for testing live in `samples/`: " +
                   ", ".join(sorted(os.listdir(sample_dir))))
    if uploaded is None:
        return

    queries_df = read_queries(uploaded)
    if queries_df.empty:
        st.warning("No queries found in that file.")
        return

    st.write("**{} queries loaded.**".format(len(queries_df)))
    st.dataframe(queries_df.head(10), use_container_width=True)

    limit = st.number_input("Answer at most this many rows", 1,
                            max(len(queries_df), 1),
                            min(len(queries_df), 100))
    if not st.button("▶️ Generate replies", type="primary"):
        return

    todo = [str(q) for q in queries_df["query"].head(int(limit))]
    progress = st.progress(0.0, text="Generating ...")
    rows, chunk = [], 16
    started = time.time()
    for i in range(0, len(todo), chunk):
        rows.extend(gen.generate_batch(todo[i: i + chunk], strategy="greedy",
                                       max_len=opts["max_len"],
                                       apply_scope_check=opts["scope_check"]))
        progress.progress(min(1.0, (i + chunk) / len(todo)),
                          text="Generating ... {}/{}".format(min(i + chunk, len(todo)),
                                                             len(todo)))
    progress.empty()

    out = pd.DataFrame([{
        "query": r["query"],
        "generated_response": r["response"],
        "in_scope": r["in_scope"],
        "unknown_word_ratio": r["oov_ratio"],
        "mean_logprob": r["avg_logprob"],
    } for r in rows])

    st.success("Answered {} queries in {:.1f}s (greedy decoding).".format(
        len(out), time.time() - started))
    c1, c2 = st.columns(2)
    c1.metric("Out-of-scope replies", int((~out["in_scope"]).sum()))
    c2.metric("Mean words per reply",
              int(out["generated_response"].str.split().str.len().mean()))

    st.dataframe(out, use_container_width=True, height=420)

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button("⬇️ Download answers as CSV", buf.getvalue(),
                       file_name="generated_responses.csv", mime="text/csv")


# --------------------------------------------------------------------------- #
def main():
    st.title("💬 Customer Support Response Generation")
    st.markdown("Encoder–decoder (seq2seq with attention) trained from scratch "
                "on a customer-support dialogue corpus — it *generates* every "
                "reply token by token rather than retrieving a template.")

    opts = sidebar()
    gen = load_generator(opts["checkpoint"])
    model_card(gen)

    chat, batch, about = st.tabs(["💬 Chat", "📄 Batch / file upload", "ℹ️ About"])
    with chat:
        chat_tab(gen, opts)
    with batch:
        batch_tab(gen, opts)
    with about:
        st.markdown(
            "**Scope.** The assistant covers {} support categories: {}.\n\n"
            "**Out-of-scope behaviour.** A query is refused when more than "
            "{:.0%} of its content words never occur in the training corpus, or "
            "when the decoder's mean per-token log-probability falls below "
            "{}. The user then gets a hand-off message instead of an invented "
            "answer.\n\n"
            "**Limitations.** Replies are drafted from patterns in historical "
            "support chats. They may contain `{{Placeholders}}` that a human "
            "agent must fill in, and they must not be sent to a customer "
            "unreviewed — the model has no access to real order, payment or "
            "account data.".format(
                len(C.IN_SCOPE_CATEGORIES),
                ", ".join(c.title() for c in C.IN_SCOPE_CATEGORIES),
                C.OOS_OOV_RATIO, C.OOS_MIN_AVG_LOGPROB))
        st.json(gen.train_config, expanded=False)


if __name__ == "__main__":
    main()
