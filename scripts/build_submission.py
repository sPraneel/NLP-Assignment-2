"""Produce every submission artefact in one go, after training has finished.

    python scripts/build_submission.py --group 7 \
        --members '[{"name":"A Kumar","id":"2023ab04567","contribution":"25%"}]'

Steps
  1. evaluate            -> reports/metrics.json, reports/manual_rating_sheet.csv
  2. capture screenshots -> reports/screenshots/*.png   (needs the app running)
  3. build notebook      -> notebooks/*.ipynb
  4. execute notebook    -> the same notebook, with outputs embedded
  5. export notebook     -> notebooks/*.html
  6. build report        -> reports/report.md, report.html, Group<N>.pdf
  7. package             -> Group<N>_Code.zip

Any step can be skipped, e.g. ``--skip evaluate,screenshots`` when the artefacts
are already up to date.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
NOTEBOOK = os.path.join("notebooks", "customer_support_response_generation.ipynb")


def jupyter_env():
    """nbconvert with an isolated config directory.

    A user-level ``~/.jupyter`` left over from an older Jupyter install (for
    example one that enabled ``jupyter_contrib_nbextensions``) makes nbconvert
    abort on import. Pointing JUPYTER_CONFIG_DIR at an empty directory keeps the
    run independent of whatever is on the machine.
    """
    env = dict(os.environ)
    env["JUPYTER_CONFIG_DIR"] = tempfile.mkdtemp(prefix="jupyter-cfg-")
    return env


def run(step, cmd, optional=False, env=None):
    print("\n" + "=" * 78)
    print("[{}]  $ {}".format(step, " ".join(cmd)))
    print("=" * 78, flush=True)
    started = time.time()
    code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if code != 0:
        msg = "step '{}' failed with exit code {}".format(step, code)
        if optional:
            print("WARNING: {} (continuing)".format(msg))
            return False
        print("ERROR: {}".format(msg))
        sys.exit(code)
    print("  done in {:.0f}s".format(time.time() - started))
    return True


def app_is_running(url="http://localhost:8501"):
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="1")
    p.add_argument("--members", default=None,
                   help='JSON list of {"name","id","contribution"}')
    p.add_argument("--eval-limit", type=int, default=400)
    p.add_argument("--skip", default="",
                   help="comma-separated: evaluate,screenshots,notebook,report,zip")
    args = p.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if "evaluate" not in skip:
        run("evaluate", [PY, "src/evaluate.py", "--limit", str(args.eval_limit),
                         "--strategy", "both"])

    if "screenshots" not in skip:
        if app_is_running():
            run("screenshots", [PY, "scripts/capture_screenshots.py"], optional=True)
        else:
            print("\nSKIPPING screenshots: the app is not running.\n"
                  "  Start it in another terminal with 'streamlit run app/app.py'\n"
                  "  and re-run with --skip evaluate,notebook,report,zip")

    if "notebook" not in skip:
        env = jupyter_env()
        run("build notebook", [PY, "scripts/build_notebook.py"])
        run("execute notebook",
            [PY, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook",
             "--inplace", "--ExecutePreprocessor.timeout=1800", NOTEBOOK], env=env)
        run("export notebook html",
            [PY, "-m", "jupyter", "nbconvert", "--to", "html", NOTEBOOK], env=env)

    if "report" not in skip:
        cmd = [PY, "scripts/build_report.py", "--group", args.group]
        if args.members:
            cmd += ["--members", args.members]
        run("build report", cmd)

    if "zip" not in skip:
        run("package", [PY, "scripts/make_submission_zip.py", "--group", args.group])

    print("\n" + "=" * 78)
    print("Submission artefacts")
    print("=" * 78)
    for rel in ["reports/Group{}.pdf".format(args.group),
                "Group{}_Code.zip".format(args.group),
                NOTEBOOK,
                NOTEBOOK.replace(".ipynb", ".html"),
                "reports/metrics.json",
                "reports/manual_rating_sheet.csv"]:
        path = os.path.join(ROOT, rel)
        mark = "OK " if os.path.exists(path) else "MISSING"
        size = " ({:.1f} MB)".format(os.path.getsize(path) / 1e6) if os.path.exists(path) else ""
        print("  {:<8} {}{}".format(mark, rel, size))
    print("\nRemaining manual steps:")
    print("  - fill in the group member table if you did not pass --members")
    print("  - complete reports/manual_rating_sheet.csv and copy the averages")
    print("    into section 5.2 of reports/report.md, then re-run build_report.py")


if __name__ == "__main__":
    main()
