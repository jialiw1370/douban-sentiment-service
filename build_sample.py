# Day 2: build_sample.py — build the labeled sample (100,000 comments)
# Copy this file to the douban-sentiment-service repo root, next to
# all_movies_with_id.csv. Do not push the project-2-day-2-en folder itself.
#
# Labels come from the Star column (Day 1 verified the distribution):
#   Star 1-2 -> negative (label 0), about 2.38M comments
#   Star 4-5 -> positive (label 1), about 4.16M comments
#   Star 3   -> neutral, dropped for this v1 binary classifier
#
# The sample is 50,000 negative + 50,000 positive = 100,000 rows,
# saved as sample.csv in the repo root. The full 2.3 GB file is read
# in chunks and never loaded into memory at once. Sampling uses
# reservoir sampling (seed 42): each comment of a class has the same
# chance of ending up in the sample, and the result is exactly
# 50,000 per class. The sample is shuffled before saving, so the two
# classes are mixed, not grouped.
#
# Why 100,000 and not all 6.5M? Training on 100k is fast on a laptop
# and plenty for a baseline — the full file stays on disk as the
# source, and the sample size is written down so the result is
# reproducible. sample.csv is gitignored — never commit it.
#
# Run (on this Mac, use python3):
#   python3 build_sample.py
# Done when it prints 50,000 negative + 50,000 positive and
# sample.csv exists in the repo root.
# Commit message: build labeled sample of 100k comments

import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "all_movies_with_id.csv"
OUTPUT_PATH = BASE_DIR / "sample.csv"
PER_CLASS = 50_000
SEED = 42
CHUNKSIZE = 200_000


def label_from_star(star: int):
    """Star 1-2 -> 0 (negative), Star 4-5 -> 1 (positive), Star 3 -> None."""
    if star in (1, 2):
        return 0
    if star in (4, 5):
        return 1
    return None


def build_sample(csv_path=CSV_PATH, output_path=OUTPUT_PATH, per_class=PER_CLASS, seed=SEED):
    rng = random.Random(seed)
    # One reservoir (list) per class. Each holds at most per_class rows.
    reservoirs = {0: [], 1: []}
    seen = {0: 0, 1: 0}  # how many usable comments of each class streamed so far
    dropped_empty = 0
    dropped_neutral = 0

    for chunk in pd.read_csv(csv_path, usecols=["Star", "Comment"], chunksize=CHUNKSIZE):
        for star, comment in zip(chunk["Star"], chunk["Comment"]):
            if not isinstance(comment, str) or not comment.strip():
                dropped_empty += 1
                continue
            label = label_from_star(int(star))
            if label is None:
                dropped_neutral += 1
                continue
            seen[label] += 1
            reservoir = reservoirs[label]
            if len(reservoir) < per_class:
                reservoir.append((comment, label))
            else:
                # Replace a random slot with probability per_class / seen.
                # That keeps every comment's chance equal (reservoir sampling).
                j = rng.randrange(seen[label])
                if j < per_class:
                    reservoir[j] = (comment, label)

    rows = reservoirs[0] + reservoirs[1]
    rng.shuffle(rows)
    sample = pd.DataFrame(rows, columns=["Comment", "label"])
    sample.to_csv(output_path, index=False)
    return sample, dropped_empty, dropped_neutral


def main():
    if not CSV_PATH.exists():
        print("all_movies_with_id.csv NOT found in the repo root.")
        print("Copy it into this folder first, then run python3 build_sample.py again.")
        return
    sample, dropped_empty, dropped_neutral = build_sample()
    print(f"Sample saved to {OUTPUT_PATH}: {len(sample)} rows")
    print(sample["label"].value_counts().to_string())
    print(f"Dropped (empty comment): {dropped_empty}")
    print(f"Dropped (Star 3, neutral): {dropped_neutral}")
    print()
    print("Example negative comments:")
    print(sample[sample["label"] == 0]["Comment"].head(3).to_string(index=False))
    print()
    print("Example positive comments:")
    print(sample[sample["label"] == 1]["Comment"].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
