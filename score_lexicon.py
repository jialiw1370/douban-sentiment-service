# Day 4 (extra): score_lexicon.py — score the OLD NRC lexicon method
# Copy this file to the douban-sentiment-service repo root, next to
# sample.csv and Chinese-Simplified-NRC-EmoLex.txt. Do not push the
# project-2-day-4-en folder itself.
#
# The 2024 notebook never trained a classifier — it counted NRC
# lexicon emotions in the top 100 comments. This script scores that
# old method the same way the Day 3 baseline was scored, on the same
# 20,000-comment test set (same seed, same split), so the two numbers
# are directly comparable:
#
#   Method: for each test comment, tokenize with jieba, count how
#   many words are in the lexicon's positive list vs. its negative
#   list. Predict positive if positive words outnumber negative
#   words, otherwise negative (ties go to negative).
#
#   Day 3 baseline for comparison: accuracy 0.7938, F1 0.7984.
#
# It also applies the notebook's cell-11 fix first: in the raw
# lexicon file the Chinese words are shifted into the 'trust'
# column (that is the column-shift bug Day 1 found). The fix moves
# them into 'Chinese-Simplified' and sets 'trust' to 0 — exactly
# what the notebook did — and the script prints whether it applied
# the fix or the file was already fixed.
#
# It also prints coverage: how many test comments contain no
# lexicon word at all. If that number is big, it explains a low
# score — the lexicon simply has no opinion on those comments.
#
# Run (on this Mac, use python3):
#   python3 score_lexicon.py
# Fast — no model training, just counting. A few minutes at most.
# Commit message: score NRC lexicon baseline

from pathlib import Path

import jieba
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "sample.csv"
LEXICON_PATH = BASE_DIR / "Chinese-Simplified-NRC-EmoLex.txt"
SEED = 42


def load_word_sets(lexicon_path):
    """Read the lexicon, apply the notebook's column-shift fix if
    needed, and return (positive_words, negative_words) as sets of
    Chinese words."""
    lexicon = pd.read_csv(lexicon_path, sep=r"\s+", header=0)
    trust_numeric = pd.to_numeric(lexicon["trust"], errors="coerce")
    if trust_numeric.isna().all():
        # Raw file: 'trust' holds Chinese words, not numbers.
        # Move them to 'Chinese-Simplified' and zero out 'trust'.
        lexicon["Chinese-Simplified"] = lexicon["trust"]
        lexicon["trust"] = 0
        print("Applied the notebook's fix: moved Chinese words from")
        print("'trust' into 'Chinese-Simplified', set trust to 0.")
    else:
        print("Lexicon file is already fixed (trust column is numeric).")
    positive = set(lexicon.loc[lexicon["positive"] == 1, "Chinese-Simplified"].dropna())
    negative = set(lexicon.loc[lexicon["negative"] == 1, "Chinese-Simplified"].dropna())
    print(f"Lexicon words: {len(positive)} positive, {len(negative)} negative")
    return positive, negative


def main():
    if not SAMPLE_PATH.exists():
        print("sample.csv NOT found in the repo root.")
        print("Run python3 build_sample.py first (Day 2), then run this again.")
        return
    if not LEXICON_PATH.exists():
        print("Chinese-Simplified-NRC-EmoLex.txt NOT found in the repo root.")
        print("Copy the lexicon file into this folder, then run again.")
        return

    positive_words, negative_words = load_word_sets(LEXICON_PATH)
    print()

    sample = pd.read_csv(SAMPLE_PATH)
    _, X_test, _, y_test = train_test_split(
        sample["Comment"], sample["label"],
        test_size=0.2, random_state=SEED, stratify=sample["label"],
    )
    print(f"Test set: {len(X_test)} comments")
    print()

    y_pred, no_hits = [], 0
    for comment in X_test:
        tokens = jieba.lcut(str(comment))
        pos = sum(1 for t in tokens if t in positive_words)
        neg = sum(1 for t in tokens if t in negative_words)
        if pos == 0 and neg == 0:
            no_hits += 1
        y_pred.append(1 if pos > neg else 0)

    print(f"Comments with no lexicon word at all: {no_hits} of {len(X_test)}")
    print()
    print(f"Lexicon method accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Lexicon method F1 (positive class): {f1_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion matrix (rows = true label, columns = predicted label):")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Day 3 baseline for comparison: accuracy 0.7938, F1 0.7984.")


if __name__ == "__main__":
    main()
