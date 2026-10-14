"""Import a real-world fake news dataset and convert it to the project's format.

This script makes it easy to use a real dataset (for example, a Kaggle "fake_and_real_news" CSV)
by normalizing its columns and saving a cleaned dataset to `dataset/fake_news_dataset.csv`.

Usage:
    python import_dataset.py --input path/to/fake_and_real_news.csv

If the CSV has a `label` column containing `FAKE` and `REAL`, they will be mapped to 0/1.

The output will overwrite `dataset/fake_news_dataset.csv` by default.
"""

import argparse
import os

import pandas as pd  # type: ignore


def normalize_label(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {'real', 'true', '1', 't', 'y', 'yes'}:
            return 1
        if v in {'fake', 'false', '0', 'f', 'n', 'no'}:
            return 0
    if pd.isna(value):
        return None
    try:
        # numeric labels
        if int(value) == 1:
            return 1
        if int(value) == 0:
            return 0
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Import a fake news dataset into the project format.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to the input CSV dataset (e.g., fake_and_real_news.csv).'
    )
    parser.add_argument(
        '--output', '-o',
        default='dataset/fake_news_dataset.csv',
        help='Path to write the normalized dataset (default: dataset/fake_news_dataset.csv).'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    # Heuristic column detection
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    if 'title' in lower_cols:
        col_map['title'] = lower_cols['title']
    elif 'headline' in lower_cols:
        col_map['title'] = lower_cols['headline']

    if 'text' in lower_cols:
        col_map['text'] = lower_cols['text']
    elif 'article' in lower_cols:
        col_map['text'] = lower_cols['article']

    if 'label' in lower_cols:
        col_map['label'] = lower_cols['label']
    elif 'truth' in lower_cols:
        col_map['label'] = lower_cols['truth']
    elif 'class' in lower_cols:
        col_map['label'] = lower_cols['class']

    missing = [k for k in ['title', 'text', 'label'] if k not in col_map]
    if missing:
        raise ValueError(
            f"Could not find required columns in input CSV. Missing: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df_clean = pd.DataFrame({
        'title': df[col_map['title']].astype(str).fillna(''),
        'text': df[col_map['text']].astype(str).fillna(''),
        'label': df[col_map['label']].apply(normalize_label),
    })

    num_na = df_clean['label'].isna().sum()
    if num_na > 0:
        print(f"Warning: {num_na} rows have unknown labels and will be dropped.")
        df_clean = df_clean[df_clean['label'].notna()]

    df_clean.to_csv(args.output, index=False)
    print(f"Saved normalized dataset to: {args.output}")
    print(f"Total rows: {len(df_clean)}")


if __name__ == '__main__':
    main()
