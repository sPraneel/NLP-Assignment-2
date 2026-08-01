"""Fetch the raw corpus so the ZIP does not have to carry an 18 MB CSV.

    python scripts/download_dataset.py

Dataset : Bitext - Customer Service Tagged Training Dataset for LLM-based
          Virtual Assistants (26,872 query/response pairs)
Source  : https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
Licence : Community Data License Agreement - Sharing, Version 1.0 (CDLA-Sharing-1.0)

If the machine has no internet access, download the CSV manually from the page
above and save it as  data/bitext_customer_support_27k.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C  # noqa: E402

URL = ("https://huggingface.co/datasets/bitext/"
       "Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/"
       "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

EXPECTED_ROWS = 26872


def main():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    if os.path.exists(C.RAW_CSV):
        print("Already present: {}".format(C.RAW_CSV))
        return

    import requests
    print("Downloading ~18 MB from Hugging Face ...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    with open(C.RAW_CSV, "wb") as fh:
        fh.write(resp.content)

    import pandas as pd
    df = pd.read_csv(C.RAW_CSV)
    print("Saved {} ({} rows, columns: {})".format(
        C.RAW_CSV, len(df), ", ".join(df.columns)))
    if len(df) != EXPECTED_ROWS:
        print("WARNING: expected {} rows, got {} - the upstream file may have "
              "changed.".format(EXPECTED_ROWS, len(df)))


if __name__ == "__main__":
    main()
