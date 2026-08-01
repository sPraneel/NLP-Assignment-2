"""Package the code submission as ``Group<N>_Code.zip`` (General Instruction 9).

    python scripts/make_submission_zip.py --group 1

The archive contains every source file and notebook, ``requirements.txt``, the
saved model / vocabulary / tokenizer artefacts, the sample input files and
``README.md``. The 18 MB raw corpus is *not* included — the archive ships
``scripts/download_dataset.py`` to fetch it instead, which the README explains.
"""

import argparse
import fnmatch
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INCLUDE = [
    "README.md",
    "requirements.txt",
    "src/*.py",
    "app/*.py",
    "scripts/*.py",
    "notebooks/*.ipynb",
    "samples/*",
    # model + tokenizer/vocabulary artefacts the evaluator needs to run the app
    "models/best_model.pt",
    "models/history.json",
    "data/processed/vocab.json",
    "data/processed/placeholder_map.json",
    "data/processed/scope_lexicon.json",
    "data/processed/preprocessing_stats.json",
    # evidence
    "reports/report.md",
    "reports/report.html",
    "reports/loss_curve.png",
    "reports/metrics.json",
    "reports/manual_rating_sheet.csv",
    "reports/train_log_transformer.txt",
    "reports/screenshots/*.png",
]

EXCLUDE = ["*/__pycache__/*", "*.pyc", ".DS_Store", "*/.ipynb_checkpoints/*"]

MAX_MB = 95.0


def matches(rel, patterns):
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch("/" + rel, p)
               for p in patterns)


def collect():
    found, missing = [], []
    for pattern in INCLUDE:
        if any(ch in pattern for ch in "*?["):
            base = os.path.dirname(pattern)
            directory = os.path.join(ROOT, base)
            if not os.path.isdir(directory):
                missing.append(pattern)
                continue
            hits = [os.path.join(base, f) for f in sorted(os.listdir(directory))
                    if fnmatch.fnmatch(os.path.join(base, f), pattern)
                    and os.path.isfile(os.path.join(ROOT, base, f))]
            if not hits:
                missing.append(pattern)
            found.extend(hits)
        else:
            if os.path.isfile(os.path.join(ROOT, pattern)):
                found.append(pattern)
            else:
                missing.append(pattern)
    return [f for f in found if not matches(f, EXCLUDE)], missing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="1")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    files, missing = collect()
    out = args.out or os.path.join(ROOT, "Group{}_Code.zip".format(args.group))

    if missing:
        print("Not found (skipped):")
        for m in missing:
            print("   {}".format(m))
        print()

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            zf.write(os.path.join(ROOT, rel), arcname=rel)

    size_mb = os.path.getsize(out) / 1e6
    print("{} files -> {} ({:.1f} MB)".format(len(files), os.path.basename(out), size_mb))
    for rel in files:
        print("   {}".format(rel))

    if size_mb > MAX_MB:
        print("\nWARNING: the archive is {:.0f} MB. If the upload limit is lower, "
              "drop models/best_model.pt and note in the README that the "
              "evaluator should run `python src/train.py`.".format(size_mb))

    required = ["README.md", "requirements.txt"]
    for r in required:
        if r not in files:
            print("\nERROR: {} is missing from the archive.".format(r))
            sys.exit(1)


if __name__ == "__main__":
    main()
