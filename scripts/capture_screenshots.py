"""Drive the running Streamlit app and save the screenshots the report needs.

    # terminal 1
    streamlit run app/app.py
    # terminal 2
    python scripts/capture_screenshots.py

Writes PNGs into reports/screenshots/:
    01_home_input_screen.png     the app as the user first sees it
    02_query_submitted.png       a query typed into the chat box
    03_generated_response.png    the generated reply plus diagnostics
    04_conversation_history.png  several turns retained on screen
    05_out_of_domain_query.png   the out-of-scope refusal
    06_batch_upload.png          the file-upload tab with results
    07_sidebar_settings.png      decoding and safety controls

Requires ``pip install playwright && playwright install chromium``.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "reports", "screenshots")

CHAT_INPUT = "textarea[data-testid='stChatInputTextArea'], div[data-testid='stChatInput'] textarea"
SETTLE = 1.2


def shot(page, name, full_page=True):
    path = os.path.join(OUT_DIR, name)
    page.screenshot(path=path, full_page=full_page)
    print("  saved {}".format(os.path.relpath(path, ROOT)))


def ask(page, question, wait=45):
    """Type a query, submit it, and wait for the reply to finish rendering."""
    box = page.locator(CHAT_INPUT).first
    box.click()
    box.fill(question)
    page.wait_for_timeout(400)
    return box


def wait_idle(page, timeout=90):
    """Streamlit shows a 'Running' status while the model decodes."""
    deadline = time.time() + timeout
    time.sleep(1.0)
    while time.time() < deadline:
        running = page.locator("[data-testid='stStatusWidget']").count()
        spinner = page.locator("[data-testid='stSpinner']").count()
        if running == 0 and spinner == 0:
            time.sleep(SETTLE)
            return
        time.sleep(0.5)
    print("  (warning: page still busy after {}s)".format(timeout))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8501")
    p.add_argument("--width", type=int, default=1440)
    p.add_argument("--height", type=int, default=1000)
    args = p.parse_args()

    from playwright.sync_api import sync_playwright

    os.makedirs(OUT_DIR, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                device_scale_factor=2)
        print("opening {}".format(args.url))
        page.goto(args.url, wait_until="networkidle", timeout=60000)
        page.wait_for_selector(CHAT_INPUT, timeout=60000)
        wait_idle(page)

        # 1. home / input screen -------------------------------------------
        shot(page, "01_home_input_screen.png")

        # 2 + 3. a query typed, then the generated reply --------------------
        q1 = "I want to cancel order 4471902"
        ask(page, q1)
        shot(page, "02_query_submitted.png")
        page.keyboard.press("Enter")
        wait_idle(page)
        shot(page, "03_generated_response.png")

        # 4. conversation history -------------------------------------------
        for q in ["How do I get a refund for my last purchase?",
                  "I forgot my password and cannot sign in"]:
            ask(page, q)
            page.keyboard.press("Enter")
            wait_idle(page)
        shot(page, "04_conversation_history.png")

        # 5. out-of-domain query --------------------------------------------
        ask(page, "What is the boiling point of water on Mars?")
        page.keyboard.press("Enter")
        wait_idle(page)
        shot(page, "05_out_of_domain_query.png")

        # 7. sidebar (captured before switching tabs) ------------------------
        page.mouse.wheel(0, -4000)
        page.wait_for_timeout(600)
        shot(page, "07_sidebar_settings.png", full_page=False)

        # 6. batch / file upload --------------------------------------------
        page.get_by_role("tab", name="Batch").click()
        wait_idle(page)
        upload = page.locator("input[type='file']").first
        upload.set_input_files(os.path.join(ROOT, "samples", "sample_queries.txt"))
        wait_idle(page)
        page.get_by_role("button", name="Generate replies").click()
        wait_idle(page, timeout=180)
        shot(page, "06_batch_upload.png")

        browser.close()

    print("\n{} screenshots in {}".format(len(os.listdir(OUT_DIR)),
                                          os.path.relpath(OUT_DIR, ROOT)))


if __name__ == "__main__":
    main()
