# Day 3: train_baseline.py — TF-IDF + Logistic Regression baseline
# Copy this file to the douban-sentiment-service repo root, next to
# sample.csv (built on Day 2: 100,000 comments, 50,000 negative + 50,000
# positive, labels from Star: 1-2 -> 0, 4-5 -> 1).
#
# What it does:
#   1. Reads sample.csv (100,000 rows).
#   2. Splits 80/20 into train/test, stratified, seed 42 — the test set
#      is 20,000 comments the model never trains on.
#   3. Tokenizes with jieba, turns comments into TF-IDF features
#      (top 50,000 words, words appearing in fewer than 2 comments
#      are ignored — they are mostly typos and one-offs).
#   4. Trains a Logistic Regression classifier on the 80,000 training
#      comments.
#   5. Prints accuracy, F1, per-class precision/recall and the
#      confusion matrix on the 20,000 test comments.
#
# Nothing is saved yet — saving the model is Day 5. This day is only
# about the first real number: how well does the simplest model do?
# For a balanced 50/50 test set, guessing would give ~50% accuracy,
# so anything clearly above that is the model actually learning.
#
# Run (on this Mac, use python3):
#   python3 train_baseline.py
# It takes a few minutes (jieba tokenizes 100k comments, then the
# model trains). Done when it prints accuracy and F1.
# Commit message: train TF-IDF and logistic regression baseline

from pathlib import Path

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "sample.csv"
MAX_FEATURES = 50_000
SEED = 42


def tokenize(text: str):
    """Split a Chinese comment into words with jieba."""
    return jieba.lcut(text)


def main():
    if not SAMPLE_PATH.exists():
        print("sample.csv NOT found in the repo root.")
        print("Run python3 build_sample.py first (Day 2), then run this again.")
        return

    sample = pd.read_csv(SAMPLE_PATH)
    print(f"Loaded sample.csv: {len(sample)} rows")
    print(sample["label"].value_counts().to_string())
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        sample["Comment"], sample["label"],
        test_size=0.2, random_state=SEED, stratify=sample["label"],
    )
    print(f"Train: {len(X_train)} rows, test: {len(X_test)} rows")
    print()

    print("Building TF-IDF features (jieba tokenizer)...")
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None,
        max_features=MAX_FEATURES, min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} words")
    print(f"Train matrix: {X_train_tfidf.shape}, test matrix: {X_test_tfidf.shape}")
    print()

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print()
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 (positive class): {f1_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion matrix (rows = true label, columns = predicted label):")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("How to read it: accuracy is the share of the 20,000 test")
    print("comments the model got right. The confusion matrix shows")
    print("where it went wrong — e.g. how many true negatives it")
    print("called positive, and how many true positives it called")
    print("negative. Day 4 looks at those mistakes in detail.")


if __name__ == "__main__":
    main()
