# Day 1: check_data.py — verify the recovered Douban data and lexicon
# Copy this file to the douban-sentiment-service repo root, next to
# all_movies_with_id.csv and the lexicon file.
# Do not push the project-2-day-1-en folder itself.
#
# Your original files are back: all_movies_with_id.csv (the 10M-row
# Douban dump the notebook read) and the NRC lexicon file. This script
# checks both without loading all 10M rows into memory at once: it
# reads the CSV in chunks and counts rows and the Star distribution
# (Star 1-5 is the label for Day 2), and it checks the lexicon file's
# columns — including whether the trust column still holds the shifted
# Chinese words (the notebook's cell 11 fix moved them out and set
# trust to 0, which is why trust was 0 in the old results).
#
# Setup (Windows: use py, not python):
#   py -m pip install -r requirements.txt
# Run:
#   py check_data.py
# Done when it prints the row count (expect about 10,310,982), the
# Star counts, and the lexicon check. The CSV and lexicon are
# gitignored — never commit them.
# Commit message: verify recovered Douban data and lexicon

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "all_movies_with_id.csv"
LEXICON_PATH = BASE_DIR / "Chinese-Simplified-NRC-EmoLex.txt"


def main():
    if not CSV_PATH.exists():
        print("all_movies_with_id.csv NOT found in the repo root.")
        print("Copy it into this folder first, then run py check_data.py again.")
        return
    print(f"File size: {CSV_PATH.stat().st_size / 1_000_000:.0f} MB")

    # Check the CSV in chunks: count rows + Star distribution.
    # Only the Star and Comment columns are read, so memory stays small.
    row_count = 0
    star_counts = {}
    for chunk in pd.read_csv(
        CSV_PATH, usecols=["Star", "Comment"], chunksize=200_000
    ):
        row_count += len(chunk)
        for star, count in chunk["Star"].value_counts().items():
            star_counts[int(star)] = star_counts.get(int(star), 0) + int(count)

    print(f"Rows: {row_count}")
    print("Star distribution (1-5):")
    for star in sorted(star_counts):
        print(f"  Star {star}: {star_counts[star]}")

    print()
    if not LEXICON_PATH.exists():
        print("Lexicon file NOT found (looked for Chinese-Simplified-NRC-EmoLex.txt).")
        print("The rebuild does not need it — labels come from Star —")
        print("but copy it here too if you want the old lexicon comparison later.")
        return
    lexicon = pd.read_csv(LEXICON_PATH, sep=r"\s+", header=0)
    print(f"Lexicon rows: {len(lexicon)}, columns: {list(lexicon.columns)}")
    # The notebook's cell 11 fix moved Chinese words out of 'trust'
    # and set it to 0. In the RAW file, 'trust' holds Chinese words
    # (the column-shift bug), so it is text, not numbers — summing it
    # would crash. Check which case this file is.
    if "trust" in lexicon.columns:
        trust_numeric = pd.to_numeric(lexicon["trust"], errors="coerce")
        if trust_numeric.isna().all():
            print("trust column holds text, not numbers: the Chinese words")
            print("are shifted into it (the column-shift bug is present in")
            print("this raw file). The notebook's fix moves them to")
            print("'Chinese-Simplified' and sets trust to 0.")
        else:
            print(
                f"trust column sum: {int(trust_numeric.sum())} "
                "(0 means the old fix is already applied)"
            )


if __name__ == "__main__":
    main()
