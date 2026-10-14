#!/usr/bin/env python3
"""
Simple test script for fake news detection with current model
"""

import pickle
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_stylometric_features_old(texts):
    """Original 6 features for compatibility with existing model"""
    features = []
    for text in texts:
        text = str(text)
        length = len(text) + 1
        num_words = len(text.split()) + 1
        features.append([
            min(length / 5000, 1.0),
            text.count('!') / length * 100,
            text.count('?') / length * 100,
            sum(1 for c in text if c.isupper()) / length * 100,
            sum(len(w) for w in text.split()) / num_words / 10.0,
            sum(1 for c in text if c in '.,!?;:') / length * 100,
        ])
    return np.array(features, dtype=np.float32)

def predict_news(text, model_package):
    """Predict using current model"""
    tfidf = model_package['tfidf']
    pac = model_package['pac']
    lr = model_package['lr']

    cleaned = clean_text(text)
    tfidf_feat = tfidf.transform([cleaned])
    style_feat = csr_matrix(extract_stylometric_features_old([cleaned]))
    X = hstack([tfidf_feat, style_feat])

    pac_proba = pac.predict_proba(X)[0]
    lr_proba = lr.predict_proba(X)[0]
    ensemble_proba = (pac_proba + lr_proba) / 2

    real_prob = float(ensemble_proba[1])

    word_count = len(cleaned.split())
    if word_count < 15:
        real_prob = 0.5 + (real_prob - 0.5) * 0.6

    prediction = 'Real' if real_prob >= 0.50 else 'Fake'
    confidence = int(round(max(real_prob, 1 - real_prob) * 100))
    confidence = min(confidence, 99)

    return prediction, confidence

def main():
    # Load current model
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model_package = pickle.load(f)

        print("📰 Fake News Detector (Current Model)")
        print("=" * 40)

        # Test examples
        test_cases = [
            "BREAKING: Scientists Discover Alien Life on Mars! NASA confirms microbial fossils found in Martian soil samples. This changes everything!",
            "Government announces new infrastructure bill worth $1.2 trillion to rebuild roads and bridges across America.",
            "Local man wins $500 million lottery jackpot for the third time this year!",
            "WHO reports decline in global COVID-19 cases as vaccination campaigns continue worldwide."
        ]

        for i, text in enumerate(test_cases, 1):
            prediction, confidence = predict_news(text, model_package)
            print(f"\n{i}. {prediction} News ({confidence}% confidence)")
            print(f"   Text: {text[:80]}...")

        print("\n" + "=" * 40)
        print("✅ Model working! Enhanced version will be available after training completes.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()