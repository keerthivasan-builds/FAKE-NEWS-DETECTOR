"""Download a public fake news dataset for training.

This script supports two modes:

1) Download a CSV from a public URL via HTTP.
2) Download from Kaggle (requires the Kaggle CLI + credentials).

The downloaded CSV will be saved as `dataset/fake_news_dataset.csv` (overwriting the existing file).

Usage:
  # Download from a public URL
  python download_dataset.py --url https://example.com/fake_news.csv

  # Download from Kaggle (requires kaggle CLI config)
  python download_dataset.py --kaggle cliveth/fake-news-detection

If you use Kaggle, ensure you have the Kaggle API credentials configured:
https://github.com/Kaggle/kaggle-api#api-credentials
"""

import argparse
import os
import shutil
import subprocess
import sys

import requests

TARGET_PATH = os.path.join('dataset', 'fake_news_dataset.csv')


def download_url(url: str, output_path: str):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        shutil.copyfileobj(resp.raw, f)

    print(f"Downloaded dataset to: {output_path}")


def download_kaggle(dataset: str, output_path: str):
    # Requires: kaggle CLI installed and configured with API credentials
    # Example dataset slug: cliveth/fake-news-detection
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download to a temporary file then move
    tmp_zip = os.path.join(os.path.dirname(output_path), 'kaggle_download.zip')

    cmd = ['kaggle', 'datasets', 'download', '-d', dataset, '-f', 'fake_and_real_news.csv', '-p', os.path.dirname(output_path), '--force']
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

    # Find the downloaded zip and unzip
    import zipfile
    with zipfile.ZipFile(tmp_zip, 'r') as z:
        z.extractall(os.path.dirname(output_path))

    # Move expected file
    possible = os.path.join(os.path.dirname(output_path), 'fake_and_real_news.csv')
    if not os.path.exists(possible):
        raise FileNotFoundError('Expected fake_and_real_news.csv inside Kaggle dataset.')

    shutil.move(possible, output_path)
    os.remove(tmp_zip)

    print(f"Downloaded Kaggle dataset to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Download a fake news dataset.')
    parser.add_argument('--url', help='Direct URL to a CSV dataset.')
    parser.add_argument('--kaggle', help='Kaggle dataset slug (e.g., cliveth/fake-news-detection).')
    parser.add_argument('--output', default=TARGET_PATH, help='Output CSV path. (default: dataset/fake_news_dataset.csv)')

    args = parser.parse_args()

    if not args.url and not args.kaggle:
        parser.error('Provide either --url or --kaggle')

    if args.url:
        download_url(args.url, args.output)
        return

    if args.kaggle:
        download_kaggle(args.kaggle, args.output)
        return


if __name__ == '__main__':
    main()
