# Day 5: test_model.py — tests for the saved model
# Copy this file to the douban-sentiment-service repo root, next to
# model.joblib and sample.csv. Do not push the project-2-day-5-en
# folder itself.
#
# Run save_model.py FIRST — these tests load model.joblib, and they
# fail if it does not exist yet. Then run:
#   python3 -m pytest test_model.py -v
#
# What the tests check:
#   1. The saved file loads and can predict.
#   2. A clearly positive comment is predicted positive.
#   3. A clearly negative comment is predicted negative.
#   4. On the same 20,000-comment test set as Days 3-4, accuracy
#      stays above 0.70 and F1 above 0.70. (Measured: 0.7938 / 0.7984.
#      The bar is set below the measured value on purpose — the test
#      should catch a broken model, not fail on small random noise.)
#
# If a test fails, read the failure message first: it usually means
# the model file is missing (run save_model.py) or stale (the sample
# changed — rerun save_model.py after rebuilding the sample).

from pathlib import Path

import pandas as pd
import pytest
from joblib import load
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
SAMPLE_PATH = BASE_DIR / "sample.csv"
SEED = 42


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), (
        "model.joblib not found — run python3 save_model.py first."
    )
    return load(MODEL_PATH)


def test_model_loads_and_predicts(model):
    preds = model.predict(["这部电影很好看"])
    assert len(preds) == 1
    assert preds[0] in (0, 1)


def test_clearly_positive_comment(model):
    assert model.predict(["这部电影太好看了，强烈推荐！"])[0] == 1


def test_clearly_negative_comment(model):
    assert model.predict(["无聊透顶，浪费时间，别看。"])[0] == 0


def test_accuracy_on_test_set(model):
    sample = pd.read_csv(SAMPLE_PATH)
    _, X_test, _, y_test = train_test_split(
        sample["Comment"], sample["label"],
        test_size=0.2, random_state=SEED, stratify=sample["label"],
    )
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.70
    assert f1_score(y_test, y_pred) > 0.70
