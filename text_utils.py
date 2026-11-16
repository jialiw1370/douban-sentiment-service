# text_utils.py — shared text helpers for the Douban sentiment service
# Copy this file to the douban-sentiment-service repo root.
#
# Why this file exists: the TF-IDF vectorizer needs a tokenizer
# function, and that function gets saved inside model.joblib. A
# function defined inside a script (like save_model.py) cannot be
# unpickled later — joblib would look for it in the *loading*
# script's __main__ and fail. And jieba.lcut itself cannot be saved
# either: it is a bound method of jieba's global tokenizer, which
# holds a thread lock that pickle refuses to save.
#
# So the tokenizer lives here, as a plain module-level function.
# joblib saves only a reference ("text_utils.tokenize"), and any
# script that loads model.joblib — test_model.py, the FastAPI app —
# just imports this module from the repo root. Keep this file in the
# repo; the saved model depends on it.

import jieba


def tokenize(text: str):
    """Split a Chinese comment into words with jieba."""
    return jieba.lcut(text)
