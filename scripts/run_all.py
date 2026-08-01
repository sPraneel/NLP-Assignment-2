"""Rebuild the whole project from a clean checkout.

    python scripts/run_all.py                 # full run
    python scripts/run_all.py --quick         # 3-epoch smoke run

Runs: download dataset -> preprocess -> train -> evaluate, then prints the
command that starts the web application.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(step, cmd):
    print("\n" + "=" * 78)
    print("STEP: {}".format(step))
    print("  $ {}".format(" ".join(cmd)))
    print("=" * 78, flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("\nFAILED at '{}' (exit code {}).".format(step, result.returncode))
        sys.exit(result.returncode)
    print("  ...done in {:.0f}s".format(time.time() - started))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="3 epochs and 100 evaluation queries, for a fast check")
    p.add_argument("--arch", default="transformer",
                   choices=["transformer", "lstm_attn"])
    args = p.parse_args()

    run("download dataset", [PY, "scripts/download_dataset.py"])
    run("preprocess", [PY, "src/preprocess.py"])

    train_cmd = [PY, "src/train.py", "--arch", args.arch]
    if args.quick:
        train_cmd += ["--epochs", "3"]
    run("train", train_cmd)

    eval_cmd = [PY, "src/evaluate.py", "--limit", "100" if args.quick else "400",
                "--strategy", "beam" if args.quick else "both"]
    run("evaluate", eval_cmd)

    print("\nAll done. Start the web application with:\n")
    print("    streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
