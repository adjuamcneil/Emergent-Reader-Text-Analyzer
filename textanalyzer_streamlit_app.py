
# streamlit_app.py
# A Streamlit wrapper that runs the existing CLI script `textanalyzer_copy_paste_entry_theta.py`
# without modifying any of its text processing functions. It feeds the expected inputs
# (role, title, proper nouns, paragraphs) to the script via stdin, captures stdout/stderr,
# and then surfaces outputs and generated files in the web UI.

import os
import io
import sys
import glob
import time
import subprocess
from pathlib import Path

import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Text Analyzer (Streamlit wrapper)", page_icon="📊", layout="wide")
st.title("📊 Emergent Reader Text Analyzer")
#st.caption("Runs your existing CLI script as a web app without changing any text-processing functions.")

# ---------------------------
# Configuration / Paths
# ---------------------------
SCRIPT_NAME = "textanalyzer_copy_paste_entry_theta.py"
SCRIPT_PATH = Path(__file__).parent / SCRIPT_NAME
REPORTS_DIR = Path(__file__).parent / "reports"

if not SCRIPT_PATH.exists():
    st.error(
        f"Cannot find `{SCRIPT_NAME}` next to this app. Place the original script in the same folder as this file.")
    st.stop()

# Helpful note about the ERTA_variables.csv dependency used by the original script
with st.expander("ℹ︎ Required data file (ERTA_variables.csv)", expanded=False):
    st.markdown(
        """
        The original script reads **`ERTA_variables.csv`** from a specific path. If that absolute path
        does not exist on your machine, please place `ERTA_variables.csv` in the current working folder
        (next to this app and your original script) **or** adjust the original script's CSV path.

        This wrapper does **not** modify any of your text processing functions; it only forwards inputs
        to your script and displays its outputs.
        """
    )

# ---------------------------
# Sidebar: Runtime options
# ---------------------------
st.sidebar.header("Runtime")
launcher = st.sidebar.radio(
    "Choose Python launcher",
    options=["py -3.12", "python"],
    index=0,
    help="Use the Windows Python launcher if you have multiple Python versions."
)

# Optional working directory for the subprocess (defaults to this file's folder)
workdir = st.sidebar.text_input(
    "Working directory for the script",
    value=str(Path(__file__).parent),
    help="Change only if your script must run from another folder to find its data files."
)

# ---------------------------
# Main inputs matching the CLI prompts in the original script
# ---------------------------
st.subheader("Inputs")
role_label_to_id = {
    "Caregiver": 1,
    "Teacher": 2,
    "Researcher": 3,
    "All Versions": 4,
}
role_label = st.selectbox("Select your report format", list(role_label_to_id.keys()), index=0)
role = role_label_to_id[role_label]

text_title = st.text_input("Title of your text", value="My Text")
proper_nouns = st.text_input(
    "Names of characters or other proper nouns (comma-separated)",
    value="",
    placeholder="e.g., Alice, Bob, Ms. Rivera"
)

st.markdown("**Paste your text below. Put a blank line between paragraphs.**")
text_data = st.text_area(
    label="Text",
    height=220,
    placeholder="Paragraph 1...\n\nParagraph 2...\n\nParagraph 3...",
)

run_btn = st.button("▶️ Run Analysis", type="primary")

# ---------------------------
# Helpers
# ---------------------------
def list_generated_files(baseline: set) -> list[Path]:
    """Return new or modified files under current folder and ./reports after running the script."""
    found = []
    # scan current dir and reports dir
    for root in [Path(workdir), Path(workdir)/"reports"]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    stat = p.stat()
                except Exception:
                    continue
                key = (str(p.resolve()), stat.st_mtime_ns, stat.st_size)
                if key not in baseline:
                    found.append(p)
    return sorted(set(found))


def snapshot_files() -> set:
    snap = set()
    for root in [Path(workdir), Path(workdir)/"reports"]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    stat = p.stat()
                except Exception:
                    continue
                snap.add((str(p.resolve()), stat.st_mtime_ns, stat.st_size))
    return snap


# ---------------------------
# Run
# ---------------------------
if run_btn:
    if not text_data.strip():
        st.warning("Please paste some text to analyze.")
        st.stop()

    # Build the exact stdin that your CLI expects:
    # 1) role (int)\n
    # 2) title\n
    # 3) proper nouns line\n
    # 4) then the multi-line text, terminated by a blank line
    # The original script stops reading paragraphs when it sees a blank line.
    import re

    # Split pasted text into paragraphs on blank lines; strip empties.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text_data.strip()) if p.strip()]

    # Build the exact input sequence the CLI expects:
    # role -> title -> proper nouns -> (each paragraph as one line) -> blank line terminator
    stdin_lines = [str(role), text_title, proper_nouns] + paragraphs + [""]
    stdin_payload = "\n".join(stdin_lines)
    # Snapshot files before running (so we can surface new outputs)
    before = snapshot_files()

    # Choose command based on launcher selection
    if launcher == "py -3.12":
        cmd = ["py", "-3.12", str(SCRIPT_PATH)]
    else:
        cmd = ["python", str(SCRIPT_PATH)]

    st.info("Running the CLI script. This may take a moment, especially the first time while spaCy loads.")
    with st.status("Executing…", expanded=True) as status:
        status.write(f"Command: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_payload.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                timeout=None,
            )
        except FileNotFoundError:
            st.error("Python launcher not found. Try switching the launcher in the sidebar.")
            st.stop()
        except subprocess.TimeoutExpired:
            st.error("The script took too long and was terminated.")
            st.stop()

        stdout_txt = proc.stdout.decode("utf-8", errors="replace")
        stderr_txt = proc.stderr.decode("utf-8", errors="replace")

        status.update(state="complete")

    # Show outputs
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Script Output (stdout)")
        st.code(stdout_txt or "<no output>", language="text")
    with col2:
        st.subheader("⚠️ Script Errors (stderr)")
        if stderr_txt.strip():
            st.code(stderr_txt, language="text")
            if "en_core_web_sm" in stderr_txt:
                st.warning("spaCy model missing? Install with: `pip install spacy` then `python -m spacy download en_core_web_sm`.\nIf you run via the Windows launcher, use `py -3.12 -m spacy download en_core_web_sm`.")
        else:
            st.success("No errors reported.")

    # Surface any files that were created/updated by the run
    after_files = list_generated_files(before)
    if after_files:
        st.subheader("📎 Generated files")
        for p in after_files:
            try:
                data = p.read_bytes()
            except Exception as e:
                st.write(f"{p} (unreadable: {e})")
                continue
            st.download_button(
                label=f"Download: {p.name}",
                data=data,
                file_name=p.name,
                mime="text/plain" if p.suffix in {".txt", ".csv", ".log"} else None,
            )
    else:
        st.info("No new files detected. If your script normally writes CSVs or text files, check its paths.")

