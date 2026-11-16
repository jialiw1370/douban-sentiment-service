# Day 4: evaluate.py — look at the baseline's mistakes
# Copy this file to the douban-sentiment-service repo root, next to
# sample.csv. Do not push the project-2-day-4-en folder itself.
#
# Day 3's baseline (TF-IDF + Logistic Regression) scored accuracy
# 0.7938 and F1 0.7984 on the 20,000-comment test set. This script
# retrains the exact same model on the exact same split (same seed,
# so the test set is identical) and then asks where it goes wrong:
#
#   1. Confusion matrix + per-class precision/recall (same as Day 3,
#      repeated here so this script stands alone).
#   2. Example errors: 8 false positives (true negative, predicted
#      positive) and 8 false negatives (true positive, predicted
#      negative), with the comment text, so you can read what kind
#      of comment fools the model.
#   3. Are errors shorter or longer than correct predictions?
#      Average comment length for correct vs. wrong predictions.
#   4. Top 10 words in the false positives and in the false
#      negatives — words that show up a lot in mistakes.
#
# What to look for: short comments with one strong word ("不错",
# "很烂") are usually easy; long mixed reviews ("前面很好看，后面
# 很无聊") and sarcasm ("真是太好看了，差点睡着") are where a
# word-counting model like TF-IDF struggles. Those patterns are the
# story for Day 7's comparison and for interviews.
#
# Run (on this Mac, use python3):
#   python3 evaluate.py
# It retrains the model, so it takes a few minutes like Day 3.
# Commit message: evaluate baseline errors

from collections import Counter
from pathlib import Path

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "sample.csv"
MAX_FEATURES = 50_000
SEED = 42
SHOW_ERRORS = 8


def tokenize(text: str):
    return jieba.lcut(text)


def show_examples(title, comments, n=SHOW_ERRORS):
    print(title)
    for i, c in enumerate(comments.head(n), 1):
        text = str(c).replace("\n", " ")
        if len(text) > 120:
            text = text[:120] + "..."
        print(f"  {i}. {text}")
    print()


def main():
    if not SAMPLE_PATH.exists():
        print("sample.csv NOT found in the repo root.")
        print("Run python3 build_sample.py first (Day 2), then run this again.")
        return

    sample = pd.read_csv(SAMPLE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        sample["Comment"], sample["label"],
        test_size=0.2, random_state=SEED, stratify=sample["label"],
    )

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None,
        max_features=MAX_FEATURES, min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion matrix (rows = true label, columns = predicted label):")
    print(confusion_matrix(y_test, y_pred))
    print()

    test = pd.DataFrame({"comment": X_test.values, "true": y_test.values, "pred": y_pred})
    false_pos = test[(test["true"] == 0) & (test["pred"] == 1)]
    false_neg = test[(test["true"] == 1) & (test["pred"] == 0)]
    correct = test[test["true"] == test["pred"]]
    print(f"False positives (true negative, called positive): {len(false_pos)}")
    print(f"False negatives (true positive, called negative): {len(false_neg)}")
    print()

    show_examples("Example FALSE POSITIVES — negative comments the model liked:", false_pos["comment"])
    show_examples("Example FALSE NEGATIVES — positive comments the model disliked:", false_neg["comment"])

    print("Average comment length (characters):")
    print(f"  correct predictions: {correct['comment'].str.len().mean():.1f}")
    print(f"  false positives:     {false_pos['comment'].str.len().mean():.1f}")
    print(f"  false negatives:     {false_neg['comment'].str.len().mean():.1f}")
    print()

    fp_words = Counter()
    for c in false_pos["comment"]:
        # Skip punctuation-only tokens (jieba returns "，" and "。" as tokens).
        fp_words.update(t for t in tokenize(str(c)) if any(ch.isalnum() for ch in t))
    fn_words = Counter()
    for c in false_neg["comment"]:
        fn_words.update(t for t in tokenize(str(c)) if any(ch.isalnum() for ch in t))
    print("Top 10 words in FALSE POSITIVES:")
    print(" ", fp_words.most_common(10))
    print("Top 10 words in FALSE NEGATIVES:")
    print(" ", fn_words.most_common(10))


if __name__ == "__main__":
    main()
