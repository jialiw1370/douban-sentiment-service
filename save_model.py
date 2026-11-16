# Day 5: save_model.py — train the baseline once and save it
# Copy this file to the douban-sentiment-service repo root, next to
# sample.csv. Do not push the project-2-day-5-en folder itself.
#
# Days 3 and 4 retrained the model every time a script ran. From now
# on the model lives in one file, model.joblib, so the API (Day 8)
# can load it instead of retraining. This script:
#
#   1. Trains the exact same baseline as Day 3 (TF-IDF with jieba
#      + Logistic Regression, same 80/20 split, same seed) but as a
#      single scikit-learn Pipeline, so tokenizing and predicting
#      stay together in one object.
#   2. Checks the saved pipeline scores the same as Day 3 on the
#      test set (accuracy 0.7938, F1 0.7984 — it should match,
#      because the data, split and settings are identical).
#   3. Saves it to model.joblib with joblib.
#
# model.joblib is gitignored (*.joblib is in .gitignore) — it is a
# build artifact, not source code. Anyone cloning the repo rebuilds
# it by running this script. Never commit it; it is tens of MB.
#
# Run (on this Mac, use python3):
#   python3 save_model.py
# Takes a few minutes like Day 3. Done when it prints the saved
# file size and the matching accuracy.
# Commit message: save trained model with joblib

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from text_utils import tokenize

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = BASE_DIR / "sample.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
MAX_FEATURES = 50_000
SEED = 42


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

    # The tokenizer comes from text_utils (see that file for why it
    # cannot be a function defined here or jieba.lcut itself).
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None,
            max_features=MAX_FEATURES, min_df=2,
        )),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    print("Training pipeline (TF-IDF + Logistic Regression)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f} (Day 3: 0.7938)")
    print(f"Test F1:       {f1_score(y_test, y_pred):.4f} (Day 3: 0.7984)")
    print()

    joblib.dump(pipeline, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved to {MODEL_PATH} ({size_mb:.1f} MB)")
    print("model.joblib is gitignored — do not commit it. Rebuild it")
    print("anytime with: python3 save_model.py")


if __name__ == "__main__":
    main()
