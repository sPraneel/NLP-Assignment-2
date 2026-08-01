"""Quick end-to-end check that a fresh setup actually works.

    python scripts/selftest.py

Verifies that the processed artefacts and the checkpoint are present, that
tokenisation round-trips, that the model generates an in-domain reply, and that
the out-of-scope gate refuses an out-of-domain query. Exits non-zero on failure,
so it can be used as a smoke test before submitting.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C  # noqa: E402

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print("  [{}] {}{}".format(status, name, (" - " + detail) if detail else ""))
    if not condition:
        failures.append(name)
    return condition


def main():
    print("1. artefacts")
    for label, path in [("vocabulary", C.VOCAB_JSON),
                        ("placeholder map", C.PLACEHOLDER_JSON),
                        ("scope lexicon", C.SCOPE_LEXICON_JSON),
                        ("train split", C.TRAIN_CSV),
                        ("test split", C.TEST_CSV)]:
        if not check(label, os.path.exists(path), os.path.relpath(path, ROOT)):
            print("\n   -> run: python src/preprocess.py")
            sys.exit(1)
    has_ckpt = check("checkpoint", os.path.exists(C.CHECKPOINT),
                     os.path.relpath(C.CHECKPOINT, ROOT))

    print("\n2. preprocessing round-trip")
    from preprocess import detokenize, preprocess_query
    from vocab import Vocabulary

    mapping = json.load(open(C.PLACEHOLDER_JSON))
    vocab = Vocabulary.load(C.VOCAB_JSON)

    tokens = preprocess_query("Heyyy!! pls cancel my ORDER 4471902 ASAP 😡", mapping)
    check("chat-speak and emoji cleaned", "please" in tokens and "😡" not in tokens,
          " ".join(tokens))
    check("reference number slotted", "<ph_order_number>" in tokens)
    # Round-trip on an in-vocabulary query. (Rare words such as "heyy" map to
    # <unk> by design, so they cannot round-trip.)
    known = preprocess_query("i want to cancel my order", mapping)
    ids = vocab.encode(known, C.MAX_SRC_LEN)
    check("encode/decode round-trip", vocab.decode(ids) == known,
          " ".join(vocab.decode(ids)))
    check("unknown words map to <unk>",
          vocab.decode(vocab.encode(["zzqqx"], C.MAX_SRC_LEN)) == ["<unk>"])
    check("detokeniser renders placeholders",
          "{{Order Number}}" in detokenize(tokens, mapping),
          detokenize(tokens, mapping))

    if not has_ckpt:
        print("\n   -> no checkpoint; run: python src/train.py")
        sys.exit(1)

    print("\n3. generation")
    from decode import ResponseGenerator
    gen = ResponseGenerator()
    check("model loaded", gen.model is not None,
          "{} on {}".format(gen.train_config.get("arch"), gen.device))

    good = gen.generate("i want to cancel my order", strategy="greedy")
    check("in-domain query accepted", good["in_scope"], good["reason"])
    check("reply is non-trivial", len(good["response"].split()) >= 15,
          "{} words".format(len(good["response"].split())))

    bad = gen.generate("who won the football world cup in 1998", strategy="greedy")
    check("out-of-domain query refused", not bad["in_scope"], bad["reason"])
    check("refusal message returned", bad["response"] == C.OOS_MESSAGE)

    beam = gen.generate("how do i get a refund", strategy="beam", beam_size=3)
    check("beam search runs", len(beam["response"].split()) >= 10)

    batch = gen.generate_batch(["where is my package", "i forgot my password"],
                               strategy="greedy")
    check("batch decoding runs", len(batch) == 2)

    print("\n" + "-" * 60)
    if failures:
        print("{} check(s) FAILED: {}".format(len(failures), ", ".join(failures)))
        sys.exit(1)
    print("All checks passed - the application is ready to run:")
    print("    streamlit run app/app.py")


if __name__ == "__main__":
    main()
