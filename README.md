# Douban Sentiment Service

Turn Douban movie comments into a sentiment classifier you can call from an
API: paste a Chinese movie comment, get back positive or negative with a
confidence score.

This rebuilds the author's 2024 Data Science III notebook
("Movie sentiment analysis", 10M+ Douban comments) as a real NLP service.
The notebook sampled comments, used jieba + TF-IDF, and counted NRC lexicon
emotions in the top 100 comments — but it never trained a classifier, so it
had no accuracy or F1. This project adds exactly that: a labeled train/test
split, a TF-IDF + Logistic Regression baseline with measured scores, a
comparison against a pre-trained Hugging Face Chinese sentiment model, and a
FastAPI service around the winner.

## Current status

- [ ] Day 1: Download the data and check it
- [ ] Day 2: Clean + sample + label (100,000 comments)
- [ ] Day 3: TF-IDF + Logistic Regression baseline
- [ ] Day 4: Evaluate properly (confusion matrix, error examples)
- [ ] Day 5: Save the model, document the scores, add a test
- [ ] Day 6: Compare with a pre-trained Hugging Face sentiment model
- [ ] Day 7: Comparison table, pick the service model
- [ ] Day 8: FastAPI `POST /predict` endpoint
- [ ] Day 9: Deploy on Render
- [ ] Day 10: README + live link

## Data

`all_movies_with_id.csv` — 10,310,982 Douban movie comments (10,269 movies),
columns include `Star` (1–5) and `Comment`. The file was lost, then found
again — copy it into the repo root and run `py check_data.py` to verify it
(row count, Star distribution, lexicon check). The same file is also public
on Hugging Face as a backup:
https://huggingface.co/datasets/dirtycomputer/douban_movie_review

The CSV and the lexicon file (`Chinese-Simplified-NRC-EmoLex.txt`) are
gitignored — never commit them. Labels come from `Star`: 1–2 = negative,
4–5 = positive, 3 = neutral (dropped for the v1 binary classifier).

## How it works (once built)

1. **Labels:** `Star` 1–2 → negative, 4–5 → positive.
2. **Baseline:** jieba tokenize → TF-IDF → Logistic Regression, 80/20 split.
3. **Comparison:** a pre-trained Hugging Face Chinese sentiment model, run
   zero-shot on the same test set.
4. **Service:** FastAPI `POST /predict` returns the label + confidence.

## Setup

```bash
py -m pip install -r requirements.txt
py check_data.py
```

On Windows, use `py` (the Python launcher). Plain `python` may open the
Microsoft Store instead of running Python.

## Tech stack

- Python + pandas + scikit-learn + jieba
- Hugging Face (dataset + pre-trained sentiment model for comparison)
- FastAPI service (from Day 8)
