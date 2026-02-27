# -*- coding: utf-8 -*-

import pandas as pd
from typing import Callable
import spacy
from collections import Counter
import os
import re

# Paste the raw text. For texts with multiple paragraphs, users will need to
# enter each paragraph separately. A new text box appears for each paragraph.
role = int(input("Select your report format: 1 for Caregiver, 2 for Teacher, 3 for Researcher 4 for All Versions: "))
text_title = input("Enter the title of your text: ")
# This allows proper nouns to be used as stop words. They will not be considered
# in word-feature analysis
proper_nouns = input("\nEnter the names of characters or other proper nouns, separated by commas: ")
custom_stop_words = {
    word.strip()
    for word in proper_nouns.split(",")
}
# If the user does not enter each paragraph separately, the analyzer will count
# the entire text as one paragraph.
print("\nIn the text box below, enter each paragraph separately. Press Enter twice (blank line) to finish: \n")
lines = []
while True:
    try:
        line = input()
    except EOFError:
        break  # In case the user ends input via EOF (Ctrl+D/Ctrl+Z)
    if line == "":  # blank line = end
        break
    lines.append(line)
text_data = "\n\n".join(lines)

# counts paragraphs, sentences within paragraphs, and words within sentences

import sys

def split_paragraphs(text_data: str):
    #text = sys.stdin.read(text_data)
    #text = text_data.strip()
    if not text_data:
       return []
    # Split on one or more blank lines (any amount of whitespace between newlines)
    paragraphs = re.split(r'\n\n', text_data)
    # Keep only non-empty paragraphs
    return [p.strip() for p in paragraphs if p.strip()]

def split_sentences(paragraph: str):

    # Normalize internal whitespace
    para = re.sub(r'[ \t]+', ' ', paragraph.strip())

    # Split on end punctuation followed by whitespace or end of string.
    # Keep punctuation with the sentence using a capturing group.
    parts = re.split(r'(?<=[.!?])\s+(?=[^\s])', para)
    # Filter out empty parts
    return [s.strip() for s in parts if s.strip()]

def count_words(sentence: str):

    tokens = re.findall(r"\b[\w'-]+\b", sentence, flags=re.UNICODE)
    return len(tokens)

# -----------------------------
# Main processing
# -----------------------------

def analyze_text(text_data: str):
    """
    Returns two DataFrames:
      1) detailed_df: one row per sentence with counts
      2) summary_df: one row per paragraph with totals
    """
    paragraphs = split_paragraphs(text_data)

    detailed_rows = []
    summary_rows = []

    for p_idx, para in enumerate(paragraphs, start=1):
        sentences = split_sentences(para)
        sentence_word_counts = []

        for s_idx, sent in enumerate(sentences, start=1):
            wc = count_words(sent)
            sentence_word_counts.append(wc)

            detailed_rows.append({
                "paragraph_id": p_idx,
                "sentence_id": s_idx,
                "sentence": sent,
                "word_count": wc
            })

        summary_rows.append({
            "paragraph_id": p_idx,
            "paragraph_text": para,
            "num_sentences": len(sentences),
            "total_words": sum(sentence_word_counts)
        })

    detailed_df = pd.DataFrame(detailed_rows,
                               columns=["paragraph_id", "sentence_id", "sentence", "word_count"])
    summary_df = pd.DataFrame(summary_rows,
                              columns=["paragraph_id", "paragraph_text", "num_sentences", "total_words"])

    return detailed_df, summary_df

# summary of text analysis
def table_summary(text_data):

    detailed, summary = analyze_text(text_data)

    print("=== Detailed (per sentence) ===")
    print(detailed)

    print("\n=== Summary (per paragraph) ===")
    print(summary)

    # Save to CSV
    detailed.to_csv(f"{text_title}.csv", index=False)
    detailed.to_csv(
        f"{text_title}_paragraph_sentence_details.csv",
        index=False,
        encoding="utf-8"
    )
    summary.to_csv(
        f"{text_title}_paragraph_summary.csv",
        index=False,
        encoding="utf-8"
    )

    # If you only want one table, use `detailed` as your result.
#code provide by Copilot 1/15/2026

# summary of text analysis
def table_summary_lite(text_data):

    detailed, summary = analyze_text(text_data)
    print("\n=== Summary (per paragraph) ===")
    print(summary)

    # If you only want one table, use `detailed` as your result.
#code provide by Copilot 1/15/2026

# This section categorizes the sentences in the text by type: simple, compound, complex, and
# compound-complex
# pip install spacy
# python -m spacy download en_core_web_sm

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit(
        "Model 'en_core_web_sm' is not installed.\n"
        "Install with:\n"
        "  pip install spacy\n"
        "  python -m spacy download en_core_web_sm"
    )


SUBORDINATORS = {
    # common subordinating conjunctions / markers
    "after", "although", "as", "because", "before", "even", "even if", "even though",
    "if", "in case", "once", "provided", "provided that", "rather than", "since",
    "so that", "than", "that", "though", "unless", "until", "when", "whenever",
    "where", "whereas", "wherever", "whether", "while"
}
DEPENDENT_CLAUSE_DEPS = {"advcl", "csubj", "csubjpass", "ccomp", "xcomp", "acl", "relcl"}

FANBOYS = {"for", "and", "nor", "but", "or", "yet", "so"}  # for readability; spaCy already links 'conj' nodes

def remove_direct_speech(text: str) -> str:
    """
    Remove text inside quotes to avoid counting quoted speech as extra clauses.
    Handles straight and curly quotes in a simple way.
    """
    # Remove “...” or "..." spans
    text = re.sub(r'“[^”]*”', "", text)
    text = re.sub(r'"[^"]*"', "", text)
    return text

def is_finite_verb(tok) -> bool:
    """Return True if token is a finite verb/aux."""
    if tok.pos_ not in ("VERB", "AUX"):
        return False
    # Finite: VerbForm=Fin (e.g., 'ask', 'ignored', 'had'); modals/aux in finite form will also show Fin
    return "Fin" in tok.morph.get("VerbForm")

def in_dependent_clause(tok) -> bool:
    """
    Return True if token is within a dependent clause span
    (i.e., it has an ancestor whose dep_ marks a subordinate/relative/adv/comp clause).
    """
    for anc in tok.ancestors:
        if anc.dep_ in DEPENDENT_CLAUSE_DEPS:
            return True
    return False

def count_independent_clauses(doc) -> int:
    """
    Count independent clauses as:
      - ROOT finite verb
      - finite verb with dep_ == 'conj' (coordinated clause)
    Excluding verbs that are within dependent clauses (advcl, ccomp, acl, relcl, etc.).
    """
    indep_heads = 0
    for tok in doc:
        if not is_finite_verb(tok):
            continue
        if tok.dep_ in ("ROOT", "conj") and not in_dependent_clause(tok):
            indep_heads += 1

    # As a safety net: if none found but sentence exists, consider at least 1 clause
    return max(indep_heads, 1 if len(doc) > 0 else 0)

def has_dependent_clause(doc) -> bool:
    """
    Detect dependent clauses via dependency labels and subordinating markers:
      - presence of advcl, ccomp, xcomp, acl, relcl, (csubj/csubjpass)
      - token with dep_ == 'mark' with a common subordinator (e.g., 'if', 'because', 'when')
    """
    # Clause-like dependency labels
    for tok in doc:
        if tok.dep_ in DEPENDENT_CLAUSE_DEPS:
            return True

    # Subordinator markers such as 'if', 'because', 'when', attached with dep_ == 'mark'
    for tok in doc:
        if tok.dep_ == "mark":
            # Compare normalized form to a set (single and multi-word forms)
            # For multi-word (e.g., "even though"), simple token check won't catch—still works well for most
            if tok.lower_ in SUBORDINATORS:
                return True

    return False

def classify_sentence_spacy(sent_text: str) -> str:
    """
    Classify one sentence string as simple / compound / complex / compound-complex.
    Removes quoted direct speech before analysis to avoid false clause counts from quotes.
    """
    stripped = remove_direct_speech(sent_text)
    # Re-parse the stripped sentence to analyze its clauses without quoted speech
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


def classify_text(
    text: str,
    nlp: "spacy.Language",
    classify_sentence_spacy: Callable[[str], str]
) -> pd.DataFrame:
    """
    Split the input text into sentences (spaCy's sentencizer),
    classify each sentence, and return a pandas DataFrame.

    Parameters
    ----------
    text : str
        The raw input text to segment and classify.
    nlp : spacy.Language
        A loaded spaCy pipeline with a sentencizer or parser that yields `doc.sents`.
    classify_sentence_spacy : Callable[[str], str]
        A function that accepts a sentence string and returns a label.

    Returns
    -------
    pd.DataFrame
        Columns:
            - sentence_id (int): 0-based index of sentence in the document
            - sentence (str): sentence text
            - label (str): predicted label for the sentence
    """
    doc = nlp(text)

    rows = []
    for i, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        label = classify_sentence_spacy(sent_text)
        rows.append(
            {
                "sentence_id": i,
                "sentence": sent_text,
                "label": label,
            }
        )

    return pd.DataFrame(rows, columns=["sentence_id", "sentence", "label"])

# This section prepares text for analysis of word features.

# Remove proper nouns from text

def remove_proper_nouns(text_data: str, custom_stop_words) -> list[str]:
    stripped_text = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)*",
        text_data,
        flags=re.IGNORECASE)
    custom_stop_words_lower = {w.lower() for w in custom_stop_words}
    npn = [word.lower() for word in stripped_text if word.lower()
    not in custom_stop_words_lower]
    return npn

no_proper_nouns = remove_proper_nouns(text_data, custom_stop_words)
# --- 1) Normalize + tokenize: lowercase, keep internal apostrophes
# (e.g., didn't, wasn't) ---

counts_npn = Counter(no_proper_nouns)
num_unique_npn = len(counts_npn)

token_table_npn = pd.DataFrame(
    sorted(counts_npn.items(), key=lambda x: (-x[1], x[0])),
    columns =["word", "count"]
)

# This script is set to a folder in my Google Drive called 'DHRI'.
# This folder is included in the project folder. All CSVs are located in that folder.


# # This codes creates the dataset of word characteristics from multiple
# databases: dELP, Chee et al.(2020), CMPD, CEFR.
#os.getcwd("")
ERTA_variables = pd.read_csv("C:/Users/amcne/OneDrive - Florida State University/FSU/Digital Incubator/DHRI/ERTA_variables.csv")
ERTA_variables.rename(columns={"NPhon": "phonemes"}, inplace=True)

#dup_mask = ERTA_variables.duplicated(subset=["word"], keep=False)
#duplicates = ERTA_variables[dup_mask].sort_values(["word"])

#if not duplicates.empty:
#    print("\nDuplicate rows (same word ):")
#    print(duplicates)
    #ERTA_variables_dedup = ERTA_variables.drop_duplicates(subset=["word"], keep="first")
    #print(len(ERTA_variables))
    #print("Rows after de-duplication:", len(ERTA_variables_dedup))


#else:
#    print("\nNo duplicates on (word).")

# Drop duplicates, keeping the first occurrence

# Use this code to find the lemmas of each word in text. This is useful for
# word lists that do not include inflected or derived forms.


# If running locally or in a fresh environment, uncomment these:
# !pip install spacy
# !python -m spacy download en_core_web_sm


# Load spaCy English model
#nlp = spacy.load("en_core_web_sm")

def lemmatize_text_spacy(text: str) -> list[str]:
    """
    Lemmatize with spaCy. Returns lowercase lemmas for alphabetic tokens
    (keeps tokens like don't, O'Neill by allowing internal apostrophes).
    """
    doc = nlp(text)
    lemmas = []
    for token in doc:
        # Keep words with optional internal apostrophes (letters only)
        if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", token.text):
            lemma = token.lemma_.lower()
            # Some older models output "-PRON-" for pronouns; fallback to token
            if lemma == "-pron-":
                lemma = token.text.lower()
            lemmas.append(lemma)
    return lemmas

#Prepares lemmas for analysis via dataframe
text = text_data

lemmas = lemmatize_text_spacy(text)
custom_stop_words_lower = {w.lower() for w in custom_stop_words}

lemmas_npn = [word.lower() for word in lemmas if word.lower() not in custom_stop_words_lower]

counts_lemmas_npn = Counter(lemmas_npn)
lemmas_unique = len(counts_lemmas_npn)
lemmas_npn_table = pd.DataFrame(
    sorted(counts_lemmas_npn.items(), key=lambda x: (-x[1], x[0])),
    columns =["word", "count"])


#print(f"\nNumber of Unique Lemmas: {lemmas_unique}\n")
#print(lemmas_npn_table)

# Use this code to find the CEFR level of the text


cefrj_subset = ERTA_variables[["word","CEFR"]]
lemma_cefrj = pd.merge(
    lemmas_npn_table, cefrj_subset,
    how="left",
    on="word"
)
lemma_cefrj['CEFR'] = pd.Categorical(
    lemma_cefrj['CEFR'],
    categories = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'],
    ordered = True
)

#print(lemma_cefrj)
not_in_cefr = lemma_cefrj[lemma_cefrj['CEFR'].isna()]
#print(not_in_cefr)

lemma_cefrj = lemma_cefrj.copy()
lemma_cefrj['CEFR'] = lemma_cefrj['CEFR'].cat.add_categories(['Unknown'])
lemma_cefrj['CEFR'] = lemma_cefrj['CEFR'].fillna('Unknown')

# Compute percentages
percentages = (
    lemma_cefrj['CEFR']
      .value_counts(normalize=True)        # proportion 0–1
      .reindex(['A1','A2','B1','B2','C1','C2','Unknown'])  # keep desired order
      .mul(100)                             # convert to %
      .round(2)                             # pretty formatting
)

#print(lemma_cefrj['CEFR'].cat.categories[
    #int(lemma_cefrj['CEFR'].cat.codes.median())
#])

# Creates a dataframe of variable values unique to the text
def merged(df):
  merged_df = pd.merge(
      df, ERTA_variables,
      how="left",             # 'inner', 'left', 'right', or 'outer'
      on="word"
  ).drop_duplicates()
  return merged_df
merged_npn = merged(token_table_npn)
#print(merged_npn)

#Word feature means and ranges
import numpy as np
#letters
letters_mean = merged_npn.Length.mean()
letters_min = merged_npn.Length.min()
letters_max = merged_npn.Length.max()
#phonemes
phonemes_mean = merged_npn.phonemes.mean()
phonemes_min = merged_npn.phonemes.min()
phonemes_max = merged_npn.phonemes.max()
#difficulty
itemDif_mean = merged_npn.difficulty.mean()
itemDif_min = merged_npn.difficulty.min()
itemDif_max = merged_npn.difficulty.max()
#syllables
syllables_mean = merged_npn.NSyll.mean()
syllables_min = merged_npn.NSyll.min()
syllables_max = merged_npn.NSyll.max()
#Standard Word Frequency Index
tasa_swfi_mean = merged_npn.tasa_sfi.mean()
tasa_swfi_min = merged_npn.tasa_sfi.min()
tasa_swfi_max = merged_npn.tasa_sfi.max()
#AOA
aoa_mean = merged_npn.AoA_Kuper.mean()
aoa_min = merged_npn.AoA_Kuper.min()
aoa_max = merged_npn.AoA_Kuper.max()
#morphemes
morphemes_mean = merged_npn.NMorph.mean()
morphemes_min = merged_npn.NMorph.min()
morphemes_max = merged_npn.NMorph.max()
#onset consistency
onsetConsist_mean = merged_npn.onset_consistency.mean()
onsetConsist_min = merged_npn.onset_consistency.min()
onsetConsist_max = merged_npn.onset_consistency.max()
#nucleus consistency
nucConsist_mean = merged_npn.nuc_consistency.mean()
nucConsist_min = merged_npn.nuc_consistency.min()
nucConsist_max = merged_npn.nuc_consistency.max()
#grade level
grade_level_mean = merged_npn.grade_eq.mean()
grade_level_min = merged_npn.grade_eq.min()
grade_level_max = merged_npn.grade_eq.max()

#This code returns the percentage of words within the first 1000 words based on
# standardized word frequency.
def first_1000(df):
  Q1 = len(merged_npn.loc[merged_npn['tasa_sfi'] >= 65, 'word']
           )/len(merged_npn)*100
  Q2 = (len(merged_npn.loc[(merged_npn['tasa_sfi'] >= 62) &
   (merged_npn['tasa_sfi'] < 65), 'word'])/len(merged_npn))*100
  Q3 = (len(merged_npn.loc[(merged_npn['tasa_sfi'] >= 60.5) &
   (merged_npn['tasa_sfi'] < 62), 'word'])/len(merged_npn))*100
  Q4 = (len(merged_npn.loc[(merged_npn['tasa_sfi'] >= 59) &
   (merged_npn['tasa_sfi'] < 60.5), 'word'])/len(merged_npn))*100
  #df.loc[df["frequency"] > 65, "word"]
  K1_plus = len(merged_npn.loc[merged_npn['tasa_sfi'] < 59, 'word']
                )/len(merged_npn)*100
  print("\nWord frequency profile based on the first 1000 words: \n")
  print(round(Q1, 2), "% Q1")
  print(round(Q2, 2), "% Q2")
  print(round(Q3, 2), "% Q3")
  print(round(Q4, 2), "% Q4")
  print(round(K1_plus, 2), "% >K1")

# This code identifies words in the text that use Level 1 letter-sound
# correspondences (i.e., alphabetic)
# Creating a library of possible phonemes for each letter
letters_to_phonemes = {
    "a": {c: 0 for c in ["AE", "EY"]},
    "b": {c: 0 for c in ["B"]},
    "c": {c: 0 for c in ["K", "S"]},
    "d": {c: 0 for c in ["D"]},
    "e": {c: 0 for c in ["EH", "IY"]},
    "f": {c: 0 for c in ["F"]},
    "g": {c: 0 for c in ["G", "JH"]},
    "h": {c: 0 for c in ["HH"]},
    "i": {c: 0 for c in ["AY", "IH"]},
    "j": {c: 0 for c in ["JH"]},
    "k": {c: 0 for c in ["K"]},
    "l": {c: 0 for c in ["L"]},
    "m": {c: 0 for c in ["M"]},
    "n": {c: 0 for c in ["N", "NG"]},
    "o": {c: 0 for c in ["AA", "OW"]},
    "p": {c: 0 for c in ["P"]},
    "q": {c: 0 for c in ["K"]},
    "r": {c: 0 for c in ["R"]},
    "s": {c: 0 for c in ["S", "Z"]},
    "t": {c: 0 for c in ["T"]},
    "u": {c: 0 for c in ["AH", "UW", "Y"]},
    "v": {c: 0 for c in ["V"]},
    "w": {c: 0 for c in ["W"]},
    "x": {c: 0 for c in ["K", "S", "Z"]},
    "y": {c: 0 for c in ["Y", "IY", "AY"]},
    "z": {c: 0 for c in ["Z"]},
}

# Load dataset
df = merged_npn
# --- Helper: strip stress digits from a phoneme like 'AE1' -> 'AE'
def strip_stress(p):
    if pd.isna(p):
        return None
    return re.sub(r'\d+', '', str(p)).strip().upper()

# --- Precompute for speed: map letter -> set of allowed phonemes (strings)
letter_allowed_sets = {ltr: set(d.keys()) for ltr, d in letters_to_phonemes.items()}

# --- Core evaluation: for each word, mark 1 if all letters have a matching phoneme, else 0
has_all_matches = []

for i in range(len(df)):
    word_raw = str(df.loc[i, "word"])
    # Normalize word: lower, keep only a-z letters to align with mapping
    word = re.sub(r'[^a-z]', '', word_raw.lower())

    # Pull phonemes for this row
    try:
        n_ph = int(df.loc[i, "phonemes"])
    except Exception:
        n_ph = 0

    # Collect this word's phonemes (stressless, uppercase)
    word_phonemes = []
    for a in range(1, n_ph + 1):
        col = f"phon{a}"
        if col in df.columns:
            p = strip_stress(df.loc[i, col])
            if p:
                word_phonemes.append(p)

    word_phoneme_set = set(word_phonemes)

    # If there are no phonemes, it cannot match
    if len(word_phoneme_set) == 0 or len(word) == 0:
        has_all_matches.append(0)
        continue

    # For each letter in the word: require at least one intersection
    all_ok = True
    for ch in word:
        if ch not in letter_allowed_sets:
            # If a letter isn't modeled, treat as failing (alternatively, you can choose to skip it)
            all_ok = False
            break
        allowed = letter_allowed_sets[ch]
        if len(allowed.intersection(word_phoneme_set)) == 0:
            all_ok = False
            break

    has_all_matches.append(1 if all_ok else 0)

# Store result and count
df["has_all_matches"] = has_all_matches
count_words_with_1 = int(df["has_all_matches"].sum())

# Create a list of words where has_all_matches == 1
words_with_all_matches = df.loc[df["has_all_matches"] == 1, "word"].tolist()
words_with_all_matches = pd.DataFrame(words_with_all_matches, columns=["word"])

# Create a list of words where has_all_matches == 0
words_with_no_match = df.loc[df["has_all_matches"] == 0, "word"].tolist()
#print("Number of words that are not Level 1 decodable:", len(words_with_no_match))
words_with_no_match = pd.DataFrame(words_with_no_match, columns=["word"])
#print(words_with_no_match)

# If you want to inspect/save:
# df.to_csv("form2_trans_with_flags.csv", index=False)

# This code creates a dataframe for all words that WERE NOT Level 1
# decodable, allowing them to be analyzed for Level 2 letter-sound
# correspondences
merged_df2 = pd.merge(
  words_with_no_match, merged_npn,
  how="left",             # 'inner', 'left', 'right', or 'outer'
  on="word"
  ).drop_duplicates()
#print(merged_df2)

# This code identifies words in the text that use Level 2 letter-sound
# correspondences (i.e., digraphs, diphthongs). Words that do not use Level 2
# LSCs are automatically categorized as Level 3.

# Creating a library of possible phonemes for each letter
letters_to_phonemes_L2 = {
    "a": {c: 0 for c in ["AE","EH", "AO","EY", "IY","OW"]},
    "b": {c: 0 for c in ["B"]},
    "c": {c: 0 for c in ["K", "S", "CH"]},
    "d": {c: 0 for c in ["D", "JH"]},
    "e": {c: 0 for c in ["EH", "IY", "ER"]},
    "f": {c: 0 for c in ["F"]},
    "g": {c: 0 for c in ["G", "JH"]},
    "h": {c: 0 for c in ["HH","CH", "DH","SH", "TH"]},
    "i": {c: 0 for c in ["AY", "IH", "ER"]},
    "j": {c: 0 for c in ["JH"]},
    "k": {c: 0 for c in ["K"]},
    "l": {c: 0 for c in ["L"]},
    "m": {c: 0 for c in ["M"]},
    "n": {c: 0 for c in ["N", "NG"]},
    "o": {c: 0 for c in ["AA", "AO", "AW","ER", "OW", "OY"]},
    "p": {c: 0 for c in ["P"]},
    "q": {c: 0 for c in ["K"]},
    "r": {c: 0 for c in ["R"]},
    "s": {c: 0 for c in ["S", "Z", "SH"]},
    "t": {c: 0 for c in ["T","CH", "DH","SH", "TH"]},
    "u": {c: 0 for c in ["UW", "AH", "Y","W","OW", "ER", "AO"]},
    "v": {c: 0 for c in ["V"]},
    "w": {c: 0 for c in ["W","OW", "AW", "AO"]},
    "x": {c: 0 for c in ["K", "S", "Z"]},
    "y": {c: 0 for c in ["Y", "IY", "AY", "EY"]},
    "z": {c: 0 for c in ["Z"]},
}

# Load dataset
df_L2 = merged_df2
df_L2 = df_L2.reset_index(drop=True)

# --- Helper: strip stress digits from a phoneme like 'AE1' -> 'AE'
def strip_stress(p):
    if pd.isna(p):
        return None
    return re.sub(r'\d+', '', str(p)).strip().upper()

# --- Precompute for speed: map letter -> set of allowed phonemes (strings)
letter_allowed_sets_L2 = {ltr: set(d.keys()) for ltr, d in letters_to_phonemes_L2.items()}

# --- Core evaluation: for each word, mark 1 if all letters have a matching phoneme, else 0
has_all_matches_L2 = []

for i in range(len(df_L2)):
    word_raw_L2 = str(df_L2.loc[i, "word"])
    # Normalize word: lower, keep only a-z letters to align with mapping
    word_L2 = re.sub(r'[^a-z]', '', word_raw_L2.lower())

    # Pull phonemes for this row
    try:
        n_ph_L2 = int(df_L2.loc[i, "phonemes"])
    except Exception:
        n_ph_L2 = 0

    # Collect this word's phonemes (stressless, uppercase)
    word_phonemes_L2 = []
    for a in range(1, n_ph_L2 + 1):
        col = f"phon{a}"
        if col in df_L2.columns:
            p = strip_stress(df_L2.loc[i, col])
            if p:
                word_phonemes_L2.append(p)

    word_phoneme_L2_set = set(word_phonemes_L2)

    # If there are no phonemes, it cannot match
    if len(word_phoneme_L2_set) == 0 or len(word_L2) == 0:
        has_all_matches_L2.append(0)
        continue

    # For each letter in the word: require at least one intersection
    all_ok = True
    for ch in word_L2:
        if ch not in letter_allowed_sets_L2:
            # If a letter isn't modeled, treat as failing (alternatively, you can choose to skip it)
            all_ok = False
            break
        allowed_L2 = letter_allowed_sets_L2[ch]
        if len(allowed_L2.intersection(word_phoneme_L2_set)) == 0:
            all_ok = False
            break

    has_all_matches_L2.append(1 if all_ok else 0)

# Store result and count
df_L2["has_all_matches_L2"] = has_all_matches_L2
count_words_with_1_L2 = int(df_L2["has_all_matches_L2"].sum())
# Create a list of words where has_all_matches == 1
words_with_all_matches_L2 = df_L2.loc[df_L2["has_all_matches_L2"] == 1, "word"].tolist()
words_with_all_matches_L2 = pd.DataFrame(words_with_all_matches_L2, columns=["word"])
# Create a list of words where has_all_matches == 0
words_with_no_match_L2 = df_L2.loc[df_L2["has_all_matches_L2"] == 0, "word"].tolist()
words_with_no_match_L2 = pd.DataFrame(words_with_no_match_L2, columns=["word"])

# If you want to inspect/save:
# df.to_csv("form2_trans_with_flags.csv", index=False)

summary_table = pd.DataFrame({"Word Features": ['Letters', 'Phonemes', 'Syllables', 'Morphemes', 'Onset Consistency', 'Vowel Consistency', 'Word Frequency Index', 'Age of Acquisition', 'Item Difficulty', 'Grade Level'],
                              "Mean": [letters_mean, phonemes_mean, syllables_mean, morphemes_mean, onsetConsist_mean, nucConsist_mean, tasa_swfi_mean, aoa_mean, itemDif_mean, grade_level_mean],
                              "Min": [letters_min, phonemes_min,syllables_min, morphemes_min, onsetConsist_min,nucConsist_min, tasa_swfi_min, aoa_min , itemDif_min, grade_level_min],
                              "Max": [letters_max, phonemes_max, syllables_max, morphemes_max, onsetConsist_max, nucConsist_max, tasa_swfi_max, aoa_max, itemDif_max, grade_level_max]})
summary_table = summary_table.round(3)

summary_table_2 = pd.DataFrame({"Word Features": ['Letters', 'Phonemes', 'Syllables', 'Word Frequency Index', 'Age of Acquisition', 'Item Difficulty', 'Grade Level'],
                              "Average": [letters_mean, phonemes_mean, syllables_mean, tasa_swfi_mean, aoa_mean, itemDif_mean, grade_level_mean],
                              "Minimum": [letters_min, phonemes_min,syllables_min, tasa_swfi_min, aoa_min , itemDif_min, grade_level_min],
                              "Maximum": [letters_max, phonemes_max, syllables_max, tasa_swfi_max, aoa_max, itemDif_max, grade_level_max]})
summary_table_2 = summary_table_2.round(2)

summary_table_3 = pd.DataFrame({"Word Features": ['Letters', 'Phonemes', 'Syllables', 'Age of Acquisition', 'Grade Level'],
                              "Average": [letters_mean, phonemes_mean, syllables_mean, aoa_mean, grade_level_mean],
                              "Minimum": [letters_min, phonemes_min,syllables_min, aoa_min , grade_level_min],
                              "Maximum": [letters_max, phonemes_max, syllables_max, aoa_max, grade_level_max]})
summary_table_3 = summary_table_3.round(1)

merged_npn.columns
merged_npn["comp_score"] = (merged_npn["Length"]*.484) + (
    merged_npn["AoA_Kuper"]*.316) + (
        merged_npn["NSyll"]*.05) + (
            merged_npn["onset_consistency"]*-0.107) + (
                merged_npn["nuc_consistency"]*-0.092) + (
                    merged_npn["coda_consistency"]*-0.044) + (
                        merged_npn["NMorph"]*-0.055) + (
                              merged_npn["tasa_sfi"]*-0.385) + (
                                  merged_npn["phonemes"]*-0.174
                              )
merged_npn["comp_score_2"] = (merged_npn["Length"]*.484) + (
        merged_npn["NSyll"]*.05) + (
            merged_npn["onset_consistency"]*-0.107) + (
                merged_npn["nuc_consistency"]*-0.092) + (
                    merged_npn["coda_consistency"]*-0.044) + (
                        merged_npn["NMorph"]*-0.055) + (
                                  merged_npn["phonemes"]*-0.174
                              )

# teachable moments

most_common = token_table_npn.head(int(round((len(token_table_npn)/10),0)))
topic = most_common.loc[merged_npn["DPoS_Brys"]=="Noun","word"].tolist()
#
#print(topic)

most_complex = (
    merged_npn
    .dropna(subset=["comp_score"])
    .sort_values("comp_score", ascending=False)
    .drop_duplicates(subset=["word"])  # optional: keep one row per word if there are duplicates
    .head(10)["word"]
    .tolist()
)
#print(most_complex)
most_complex_2 = (
    merged_npn
    .dropna(subset=["comp_score_2"])
    .sort_values("comp_score_2", ascending=False)
    .drop_duplicates(subset=["word"])  # optional: keep one row per word if there are duplicates
    .head(10)["word"]
    .tolist()
)
#print(most_complex_2)
##
#
# --- your existing report functions (unchanged, except one small f-string fix) ---

def researcher_report():
    print("\nResearcher Report")
    print(text_title)
    print("\n")
    print(text_data)
    print("\n")
    table_summary(text_data)

    # Correctly call classify_text with (text, nlp, function) and iterate the DataFrame
    classified_df = classify_text(text_data, nlp, classify_sentence_spacy)
    print(classified_df)

    print(f"\n{lemma_cefrj['CEFR'].value_counts()}\n")
    print("\nPercentage of words by CEFR level:")
    print(percentages.to_string())
    # Fixed: Added 'f' prefix to allow the variables to display
    print(f"\nCEFR Mode: {lemma_cefrj['CEFR'].mode()[0]}")
    print(f"\nCEFR Median: {lemma_cefrj['CEFR'].cat.categories[int(lemma_cefrj['CEFR'].cat.codes.median())]}")
    print(f"\nWORD-LEVEL ANALYSES DO NOT INCLUDE PROPER NOUNS")
    print(f"\nNumber of unique words: {num_unique_npn}\n")
    print(token_table_npn.to_string(index=False))
    print("\nWord-Level Features\n")
    print(summary_table)

    # Fixed: first_1000 already prints, so we don't need to wrap it in another print()
    first_1000(merged_npn)
    print("\nLevel 1 decodability: alphabetic letter-sound correspondences, including regular and 'soft' consonants and short and long vowels")
    print("\nNumber of Level 1 decodable words (types):", count_words_with_1)
    print("Percentage of word list:", round((count_words_with_1/len(merged_npn))*100, 2))
    print("\nLevel 2 decodability: digraphs, trigraphs, diphthongs, single vowel+R, and vowel teams")
    print("\nNumber of Level 2 decodable words (types):", count_words_with_1_L2)
    print("Percentage of word list:", round((count_words_with_1_L2/len(merged_npn))*100, 2))
    print("\nLevel 3 decodability: variant vowel sounds, including schwa, vowel teams+R, and irregular vowel teams")
    print("\nNumber of Level 3 decodable words (types):", (len(words_with_no_match_L2)))
    print("Percentage of word list:", round((len(words_with_no_match_L2)/len(merged_npn))*100, 2))

def teacher_report():
    print("\nTeacher Report")
    print("\n")
    print(text_title)
    proper_nouns_list = [noun.strip() for noun in proper_nouns.split(',')]
    processed_nouns = []
    for noun in proper_nouns_list:
        if noun.endswith("'s"):
            processed_nouns.append(noun[:-2])
        elif noun.endswith("'"):
            processed_nouns.append(noun[:-1])
        else:
            processed_nouns.append(noun)
    unique_proper_nouns = list(dict.fromkeys(processed_nouns))

    print("\nTopic(s): ", ", ".join(topic + unique_proper_nouns))
    print("\n")
    #print(text_data)
    print("\n")
    # Correctly call classify_text with (text, nlp, function) and iterate the DataFrame
    classified_df = classify_text(text_data, nlp, classify_sentence_spacy)
    print(classified_df)
    table_summary_lite(text_data)

    print(f"\nFor ESL Instruction:")
    # Fixed: add f so the value counts are displayed instead of the literal braces
    print(f"\n{lemma_cefrj['CEFR'].value_counts()}\n")
    print("\nPercentage of words by CEFR level:")
    print(percentages.to_string())

    # Fixed: Added 'f' prefix to allow the variables to display
    print(f"\nCEFR Mode: {lemma_cefrj['CEFR'].mode()[0]}")
    print(f"\nCEFR Median: {lemma_cefrj['CEFR'].cat.categories[int(lemma_cefrj['CEFR'].cat.codes.median())]}")
    print(f"\nWORD-LEVEL ANALYSES DO NOT INCLUDE PROPER NOUNS\n")
    print(f"\nNumber of unique words: {num_unique_npn}\n")
    print(token_table_npn.to_string(index=False))
    print("\n")
    print(summary_table_2)

    #first_1000(merged_npn)
    print("\nWords to preteach:", ", ".join(most_complex_2))
    print("\nLevel 1: words that can be decoded using knowledge of the alphabet")
    print("\nNumber of Level 1 decodable words (types):", count_words_with_1)
    print("Percentage of Level 1 decodable words (types):", round((count_words_with_1/len(merged_npn))*100, 2))
    print("Some of the Level 1 words in this text: ", "  ".join(words_with_all_matches['word'].head(10).astype(str)))
    print("\nLevel 2: words that include digraphs, trigraphs, diphthongs, single vowel+R, and vowel teams")
    print("\nNumber of Level 2 decodable words (types):", count_words_with_1_L2)
    print("Percentage of Level 2 decodable words (types):", round((count_words_with_1_L2/len(merged_npn))*100, 2))
    print("Some of the Level 2 words in this text: ","  ".join(words_with_all_matches_L2['word'].head(10).astype(str)))
    print("\nLevel 3: words that include irregular vowel sounds, schwa, vowel teams+R, and irregular vowel teams")
    print("\nNumber Level 3 decodable words (types):", (len(words_with_no_match_L2)))
    print("Percentage of Level 3 decodable words (types):", round((len(words_with_no_match_L2)/len(merged_npn))*100, 2))
    print("Some of the Level 3 words in this text: ", "  ".join(words_with_no_match_L2['word'].head(10).astype(str)))

def caregiver_report():
    print("\nCaregiver Report")
    print("\n")
    print(text_title)
    proper_nouns_list = [noun.strip() for noun in proper_nouns.split(',')]
    processed_nouns = []
    for noun in proper_nouns_list:
        if noun.endswith("'s"):
            processed_nouns.append(noun[:-2])
        elif noun.endswith("'"):
            processed_nouns.append(noun[:-1])
        else:
            processed_nouns.append(noun)
    unique_proper_nouns = list(dict.fromkeys(processed_nouns))

    print("\nTopic(s): ", ", ".join(topic + unique_proper_nouns))
    print("\n")
    print(text_data)
    print("\n")
    # Correctly call classify_text with (text, nlp, function) and iterate the DataFrame
    classified_df = classify_text(text_data, nlp, classify_sentence_spacy)
    #print(classified_df)
    table_summary_lite(text_data)
    print("\nWord-Level Features - PROPER NOUNS NOT INCLUDED")
    print(f"\nNumber of unique words: {num_unique_npn}\n")
    print(token_table_npn.to_string(index=False))
    print("\n")
    print(summary_table_3)

    #first_1000(merged_npn)
    print("\nWords to review before reading: ", ", ".join(most_complex_2))
    print("\nLevel 1: words that can be decoded using knowledge of the alphabet")
    print("Percentage of Level 1 words:", round((count_words_with_1/len(merged_npn))*100, 2))
    print("Some of the Level 1 words in this text: ", "  ".join(words_with_all_matches['word'].head(5).astype(str)))
    print("\nLevel 2: words that include digraphs, trigraphs, diphthongs, single vowel+R, and vowel teams")
    print("Percentage of Level 2 words:", round((count_words_with_1_L2/len(merged_npn))*100, 2))
    print("Some of the Level 2 words in this text: ","  ".join(words_with_all_matches_L2['word'].head(5).astype(str)))
    print("\nLevel 3: words that include irregular vowel sounds, schwa, vowel teams+R, and irregular vowel teams")
    print("Percentage of Level 3 words:", round((len(words_with_no_match_L2)/len(merged_npn))*100, 2))
    print("Some of the Level 3 words in this text: ", "  ".join(words_with_no_match_L2['word'].head(5).astype(str)))


# --- NEW: write each report's printed output directly to a text file ---
# --- NEW: write each report's printed output to a text file **and** echo to screen ---

from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
import sys
import re

OUTPUT_DIR = Path("reports")

class Tee:
    """
    A simple stream that duplicates writes to multiple streams.
    Here we use it to send stdout to both the file and the console.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

def _slugify(s: str) -> str:
    """Make a filesystem-friendly slug (lowercase, hyphens)."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", s).strip("-").lower()

def _write_report(func, filename_stem: str) -> Path:
    """
    Runs `func()` while teeing everything it prints to BOTH:
      - a UTF-8 text file under ./reports
      - the console (sys.stdout)
    Returns the Path to the written file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Try to enrich the filename with the text title when available
    try:
        title_slug = _slugify(text_title) if text_title else ""
    except NameError:
        title_slug = ""

    suffix = f"_{title_slug}" if title_slug else ""
    path = OUTPUT_DIR / f"{filename_stem}{suffix}_{ts}.txt"

    with open(path, "w", encoding="utf-8") as fh:
        # Tee: write to file AND to the console
        tee = Tee(sys.stdout, fh)
        with redirect_stdout(tee):
            func()

    return path

def write_reports_for_role(role: int):
    """
    role mapping:
      1 = caregiver
      2 = teacher
      3 = researcher
      4 = all three
    Writes each selected report to a text file and returns a list of Paths.
    """
    written = []
    if role == 3:
        written.append(_write_report(researcher_report, "researcher_report"))
    elif role == 2:
        written.append(_write_report(teacher_report, "teacher_report"))
    elif role == 1:
        written.append(_write_report(caregiver_report, "caregiver_report"))
    elif role == 4:
        written.append(_write_report(caregiver_report, "caregiver_report"))
        written.append(_write_report(teacher_report, "teacher_report"))
        written.append(_write_report(researcher_report, "researcher_report"))
    else:
        # Unknown role; nothing to write
        return []

    return written


# --- call this instead of directly calling the report functions ---

if __name__ == "__main__":
    paths = write_reports_for_role(role)
    for p in paths:
        print(f"Saved: {p}")  # console confirmation