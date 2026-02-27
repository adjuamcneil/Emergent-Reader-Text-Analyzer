import streamlit as st
import pandas as pd
import spacy
import re
from collections import Counter
from pathlib import Path
from typing import Callable
streamlit
pandas
spacy
numpy
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz

# --- Configuration & Setup ---
st.set_page_config(page_title="DHRI Text Analyzer", layout="wide")


@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")


nlp = load_nlp()

# Path setup for your data files
script_dir = Path(__file__).parent
base_path = script_dir / "DHRI"


# --- Helper Functions (Your Original Logic) ---

def split_paragraphs(text_data: str):
    if not text_data: return []
    paragraphs = re.split(r'\n\n', text_data)
    return [p.strip() for p in paragraphs if p.strip()]


def split_sentences(paragraph: str):
    para = re.sub(r'[ \t]+', ' ', paragraph.strip())
    parts = re.split(r'(?<=[.!?])\s+(?=[^\s])', para)
    return [s.strip() for s in parts if s.strip()]


def count_words(sentence: str):
    tokens = re.findall(r"\b[\w'-]+\b", sentence, flags=re.UNICODE)
    return len(tokens)


def remove_direct_speech(text: str) -> str:
    text = re.sub(r'“[^”]*”', "", text)
    text = re.sub(r'"[^"]*"', "", text)
    return text


def is_finite_verb(tok) -> bool:
    if tok.pos_ not in ("VERB", "AUX"): return False
    return "Fin" in tok.morph.get("VerbForm")


DEPENDENT_CLAUSE_DEPS = {"advcl", "csubj", "csubjpass", "ccomp", "xcomp", "acl", "relcl"}
SUBORDINATORS = {"after", "although", "as", "because", "before", "if", "since", "unless", "until", "when", "while"}


def in_dependent_clause(tok) -> bool:
    for anc in tok.ancestors:
        if anc.dep_ in DEPENDENT_CLAUSE_DEPS: return True
    return False


def count_independent_clauses(doc) -> int:
    indep_heads = 0
    for tok in doc:
        if not is_finite_verb(tok): continue
        if tok.dep_ in ("ROOT", "conj") and not in_dependent_clause(tok):
            indep_heads += 1
    return max(indep_heads, 1 if len(doc) > 0 else 0)


def has_dependent_clause(doc) -> bool:
    for tok in doc:
        if tok.dep_ in DEPENDENT_CLAUSE_DEPS or (tok.dep_ == "mark" and tok.lower_ in SUBORDINATORS):
            return True
    return False


def classify_sentence_spacy(sent_text: str) -> str:
    stripped = remove_direct_speech(sent_text)
    doc = nlp(stripped)
    indep = count_independent_clauses(doc)
    dep = has_dependent_clause(doc)
    if indep >= 2 and dep:
        return "compound-complex"
    elif indep >= 2:
        return "compound"
    elif dep:
        return "complex"
    else:
        return "simple"


def remove_proper_nouns(text_data: str, custom_stop_words) -> list[str]:
    stripped_text = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", text_data, flags=re.IGNORECASE)
    custom_stop_words_lower = {w.lower() for w in custom_stop_words}
    return [word.lower() for word in stripped_text if word.lower() not in custom_stop_words_lower]


def lemmatize_text_spacy(text: str) -> list[str]:
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token.text):
            lemma = token.lemma_.lower()
            if lemma == "-pron-": lemma = token.text.lower()
            lemmas.append(lemma)
    return lemmas


# --- Sidebar Inputs ---
st.sidebar.header("Analysis Settings")
text_title = st.sidebar.text_input("Text Title", value="My Analysis")
proper_nouns_input = st.sidebar.text_input("Proper Nouns (comma separated)", value="")
custom_stop_words = {word.strip() for word in proper_nouns_input.split(",") if word.strip()}

# --- Main UI ---
st.title("DHRI Text Analysis Tool")
text_data = st.text_area("Paste your text here (separate paragraphs with a blank line):", height=300)

if text_data:
    # 1. Basic Analysis
    paragraphs = split_paragraphs(text_data)
    detailed_rows = []
    summary_rows = []

    for p_idx, para in enumerate(paragraphs, start=1):
        sentences = split_sentences(para)
        sentence_word_counts = []
        for s_idx, sent in enumerate(sentences, start=1):
            wc = count_words(sent)
            sentence_word_counts.append(wc)
            detailed_rows.append({"paragraph_id": p_idx, "sentence_id": s_idx, "sentence": sent, "word_count": wc,
                                  "label": classify_sentence_spacy(sent)})
        summary_rows.append(
            {"paragraph_id": p_idx, "num_sentences": len(sentences), "total_words": sum(sentence_word_counts)})

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)

    # 2. Word Level Data Loading
    try:
        dELP = pd.read_csv(base_path / "full_item_level_database.csv")
        dELP_subset = dELP[
            ["word_lowercase", "difficulty", "grade_eq", "tasa_sfi", "AoA_Kuper", "DPoS_Brys", "Conc_Brys", "NMorph",
             "Nmeanings_Websters", "Length", "NPhon", "NSyll"]].rename(
            columns={"word_lowercase": "word", "NPhon": "phonemes"})

        consistency_df = pd.read_csv(base_path / "consistency.csv")
        consistency_df["onsetnuc_consistency"] = consistency_df["ff_all_on"].fillna(consistency_df["ff_1_on"])
        consistency_df["nuc_consistency"] = consistency_df["ff_all_n"].fillna(consistency_df["ff_1_n"])
        consistency_df["onset_consistency"] = consistency_df["ff_all_o"].fillna(consistency_df["ff_1_o"])
        consistency_df["coda_consistency"] = consistency_df["ff_all_c"].fillna(consistency_df["ff_1_c"])

        cmpd = pd.read_csv(base_path / "CMPD.csv", dtype="string")

        variables = pd.merge(dELP_subset, consistency_df[
            ["word", "onsetnuc_consistency", "nuc_consistency", "onset_consistency", "coda_consistency"]], on="word",
                             how="inner")
        variables = variables.merge(cmpd, on="word", how="inner").drop_duplicates(subset=["word"])

        # CEFR
        cefrj = pd.read_csv(base_path / "cefrj.csv").rename(columns={"headword": "word"})
        cefrj["word"] = cefrj["word"].str.replace(r"/.*$", "", regex=True)
    except FileNotFoundError as e:
        st.error(f"Missing data file in DHRI folder: {e}")
        st.stop()

    # 3. Processing Tables
    no_proper_nouns = remove_proper_nouns(text_data, custom_stop_words)
    token_table_npn = pd.DataFrame(Counter(no_proper_nouns).items(), columns=["word", "count"]).sort_values(by="count",
                                                                                                            ascending=False)
    merged_npn = pd.merge(token_table_npn, variables, on="word", how="left").drop_duplicates()

    # Complex Scores
    merged_npn["comp_score_2"] = (merged_npn["Length"] * .484) + (merged_npn["NSyll"] * .05) + (
                merged_npn["onset_consistency"] * -0.107) + (merged_npn["nuc_consistency"] * -0.092) + (
                                             merged_npn["phonemes"] * -0.174)
    most_complex_words = \
    merged_npn.dropna(subset=["comp_score_2"]).sort_values("comp_score_2", ascending=False).head(10)["word"].tolist()

    # --- Tabs for Reports ---
    tab1, tab2, tab3 = st.tabs(["Caregiver Report", "Teacher Report", "Researcher Report"])

    with tab1:
        st.header("Caregiver Report")
        st.write(f"**Title:** {text_title}")
        st.subheader("Text Summary")
        st.table(summary_df)
        st.write(f"**Words to review before reading:** {', '.join(most_complex_words)}")
        # Add your Level 1/2/3 logic here as st.metric or st.write

    with tab2:
        st.header("Teacher Report")
        st.dataframe(detailed_df[["paragraph_id", "sentence", "label", "word_count"]])
        st.subheader("ESL Insights")
        # CEFR logic here
        st.write(f"**Pre-teach words:** {', '.join(most_complex_words)}")

    with tab3:
        st.header("Researcher Report")
        st.subheader("Sentence Classification")
        st.dataframe(detailed_df)
        st.subheader("Word Level Metrics")
        st.write(merged_npn.describe())

        # Download Button
        csv = detailed_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Detailed CSV", csv, f"{text_title}_analysis.csv", "text/csv")

else:

    st.info("Please enter text to begin analysis.")

