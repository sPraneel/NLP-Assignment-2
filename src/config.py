"""Central configuration for the Customer Support Response Generation project.

Every path is derived from the repository root so the project can be cloned and
run from any directory (including the BITS OSHA virtual lab) without edits.
"""

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
SAMPLE_DIR = os.path.join(ROOT_DIR, "samples")

RAW_CSV = os.path.join(RAW_DIR, "bitext_customer_support_27k.csv")

TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
VALID_CSV = os.path.join(PROCESSED_DIR, "valid.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")
VOCAB_JSON = os.path.join(PROCESSED_DIR, "vocab.json")
PLACEHOLDER_JSON = os.path.join(PROCESSED_DIR, "placeholder_map.json")
SCOPE_LEXICON_JSON = os.path.join(PROCESSED_DIR, "scope_lexicon.json")
STATS_JSON = os.path.join(PROCESSED_DIR, "preprocessing_stats.json")

CHECKPOINT = os.path.join(MODEL_DIR, "best_model.pt")
HISTORY_JSON = os.path.join(MODEL_DIR, "history.json")
LOSS_CURVE_PNG = os.path.join(REPORT_DIR, "loss_curve.png")
METRICS_JSON = os.path.join(REPORT_DIR, "metrics.json")

# --------------------------------------------------------------------------- #
# Special tokens
# --------------------------------------------------------------------------- #
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
MAX_SRC_LEN = 32          # customer query, in tokens (incl. <sos>/<eos>)
MAX_TGT_LEN = 120         # agent reply, in tokens (incl. <sos>/<eos>)
MIN_FREQ = 2              # a word must appear at least this often to enter vocab
SPLIT_RATIOS = (0.80, 0.10, 0.10)   # train / valid / test
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Model (defaults; overridable from the CLI in train.py)
# --------------------------------------------------------------------------- #
ARCH = "transformer"      # "transformer" | "lstm_attn"

# Transformer
D_MODEL = 256
N_HEADS = 4
N_ENC_LAYERS = 3
N_DEC_LAYERS = 3
FFN_DIM = 512
DROPOUT = 0.1

# LSTM + Bahdanau attention
EMB_DIM = 256
HIDDEN_DIM = 512
LSTM_LAYERS = 1

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
CLIP_NORM = 1.0
PATIENCE = 4              # early stopping on validation loss
WARMUP_STEPS = 400

# --------------------------------------------------------------------------- #
# Decoding / inference
# --------------------------------------------------------------------------- #
DECODE_STRATEGY = "beam"  # "greedy" | "beam"
BEAM_SIZE = 3
LENGTH_PENALTY = 0.7
MAX_DECODE_LEN = 120
NO_REPEAT_NGRAM = 3

# --------------------------------------------------------------------------- #
# Out-of-scope (OOS) detection thresholds
# --------------------------------------------------------------------------- #
# A query is out of scope when too many of its *content* words are unknown to
# the training queries, or when the decoder itself is very unsure.
#
# The OOV ratio is measured against the query side of the training split only
# (data/processed/scope_lexicon.json). Measuring it against the full model
# vocabulary does not work: that vocabulary also contains every word of every
# agent reply, which makes almost any English sentence look familiar.
OOS_OOV_RATIO = 0.50          # >50% of content words unseen  -> out of scope
OOS_MIN_AVG_LOGPROB = -1.50   # mean per-token log-prob below this -> unsure
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
