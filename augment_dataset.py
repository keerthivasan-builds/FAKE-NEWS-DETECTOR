# augment_dataset.py

"""Create a larger training dataset by generating synthetic variants.

This script reads the base dataset at `dataset/fake_news_dataset.csv` and
creates an expanded version with additional synthetic rows.

The goal is to boost model accuracy in small-demo scenarios by providing
more examples during training.

Usage:
    python augment_dataset.py

This will create (or overwrite) `dataset/fake_news_dataset_augmented.csv`.
"""

import random
import pandas as pd

SOURCE_PATH = 'dataset/fake_news_dataset.csv'
OUTPUT_PATH = 'dataset/fake_news_dataset_augmented.csv'

# A few short filler phrases we can insert to create variation.
EXTRA_PHRASES = [
    'Experts say',
    'According to reports,',
    'In an exclusive interview,',
    'Breaking:',
    'New study shows',
    'However,',
    'Some claim',
    'Officials warn that',
]


def augment_text(text: str) -> str:
    """Make a small random change to a sentence to create a new variation."""
    if not text or len(text.split()) < 5:
        return text

    words = text.split()
    # Randomly insert a phrase at the beginning 50% of the time
    if random.random() < 0.5:
        prefix = random.choice(EXTRA_PHRASES)
        return f"{prefix} {text}"

    # Otherwise, swap two random words to create a slight variation
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return ' '.join(words)


def main():
    df = pd.read_csv(SOURCE_PATH)
    augmented = [df]

    # Create N variants of each row
    variants_per_row = 10
    for _ in range(variants_per_row):
        copy = df.copy()
        copy['title'] = copy['title'].apply(augment_text)
        copy['text'] = copy['text'].apply(augment_text)
        augmented.append(copy)

    result = pd.concat(augmented, ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Augmented dataset written to: {OUTPUT_PATH}")
    print(f"Total examples: {len(result)}")


if __name__ == '__main__':
    main()
