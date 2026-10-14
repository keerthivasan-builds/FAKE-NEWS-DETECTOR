import pandas as pd
import os

def download_and_format():
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
    print(f"Downloading dataset from {url}...")
    
    # Read CSV directly from URL
    df = pd.read_csv(url)
    
    # Standard dataset often contains columns: 'Unnamed: 0', 'title', 'text', 'label'
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    print(f"Original shape: {df.shape}")
    
    # Map FAKE/REAL strings to 0/1 integers for compatibility with app.py exact match logic
    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    
    # Drop NAs
    df = df.dropna(subset=['title', 'text', 'label'])
    
    # Take a subsample of 5000 rows to speed up model training since we just want proof of concept, not a 3-hour local training sweep.
    # The dataset has 6300 rows so 5000 is plenty enough to recognize basic vocabulary.
    df = df.sample(n=min(5000, len(df)), random_state=42)
    
    print(f"Processed shape: {df.shape}")
    print(df['label'].value_counts())
    
    os.makedirs('dataset', exist_ok=True)
    out_path = 'dataset/fake_news_dataset.csv'
    df.to_csv(out_path, index=False)
    
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    download_and_format()
