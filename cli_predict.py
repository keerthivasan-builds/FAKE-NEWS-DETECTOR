#!/usr/bin/env python3
"""
Command-line interface for fake news detection
"""

import pickle
import os
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

def extract_stylometric_features(texts):
    """Extract the same stylometric features used during training."""
    features = []
    for text in texts:
        text = str(text)
        length = len(text) + 1
        num_words = len(text.split()) + 1
        num_sentences = len([s for s in text.split('.') if s.strip()]) + 1

        words = text.split()
        short_words = sum(1 for w in words if len(w) <= 3) / num_words
        long_words = sum(1 for w in words if len(w) >= 8) / num_words
        avg_sentence_len = num_words / num_sentences
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1) / num_words
        title_case_words = sum(1 for w in words if w.istitle()) / num_words
        unique_punct = len(set(c for c in text if c in '.,!?;:()[]{}')) / length

        features.append([
            min(length / 5000, 1.0),
            text.count('!') / length * 100,
            text.count('?') / length * 100,
            sum(1 for c in text if c.isupper()) / length * 100,
            sum(len(w) for w in text.split()) / num_words / 10.0,
            sum(1 for c in text if c in '.,!?;:') / length * 100,
            short_words,
            long_words,
            min(avg_sentence_len / 50, 1.0),
            all_caps_words,
            title_case_words,
            unique_punct * 100,
        ])
    return np.array(features, dtype=np.float32)

def predict_news(text, model_package):
    """Predict if news is real or fake"""
    cleaned = clean_text(text)

    if model_package.get('type') == 'ensemble_v4':
        tfidf_word = model_package['tfidf_word']
        tfidf_char = model_package['tfidf_char']
        models = model_package.get('models', {})
        weights = model_package.get('weights', {})
        threshold = float(model_package.get('decision_threshold', 0.5))

        word_feat = tfidf_word.transform([cleaned])
        char_feat = tfidf_char.transform([cleaned])
        style_feat = csr_matrix(extract_stylometric_features([cleaned]))
        X = hstack([word_feat, char_feat, style_feat])

        probs = []
        raw_weights = []
        for name in ['pac', 'lr', 'svc']:
            model_obj = models.get(name)
            if model_obj is not None:
                probs.append(model_obj.predict_proba(X)[0])
                raw_weights.append(max(float(weights.get(name, 1.0)), 0.0))

        if not probs:
            raise ValueError('No valid sub-models found in ensemble_v4 package.')

        if sum(raw_weights) <= 0:
            raw_weights = [1.0] * len(raw_weights)

        total = float(sum(raw_weights))
        ensemble_proba = np.zeros(2, dtype=np.float64)
        for i, p in enumerate(probs):
            ensemble_proba += (raw_weights[i] / total) * p

        real_prob = float(ensemble_proba[1])
        prediction = 'Real' if real_prob >= threshold else 'Fake'
    else:
        tfidf = model_package['tfidf']
        pac = model_package['pac']
        lr = model_package['lr']
        rf = model_package.get('rf')

        tfidf_feat = tfidf.transform([cleaned])
        style_feat = csr_matrix(extract_stylometric_features([cleaned]))
        X = hstack([tfidf_feat, style_feat])

        pac_proba = pac.predict_proba(X)[0]
        lr_proba = lr.predict_proba(X)[0]

        if rf is not None:
            rf_proba = rf.predict_proba(X)[0]
            ensemble_proba = (pac_proba + lr_proba + rf_proba) / 3
        else:
            ensemble_proba = (pac_proba + lr_proba) / 2

        real_prob = float(ensemble_proba[1])
        prediction = 'Real' if real_prob >= 0.50 else 'Fake'

    # Very short text → less reliable
    word_count = len(cleaned.split())
    if word_count < 15:
        real_prob = 0.5 + (real_prob - 0.5) * 0.6

    confidence = int(round(max(real_prob, 1 - real_prob) * 100))
    confidence = min(confidence, 99)

    return prediction, confidence

def main():
    # Load model
    model_path = 'fake_news_model.pkl'
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please run 'python model.py' first.")
        return

    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)

        if not isinstance(model_package, dict):
            print("❌ Invalid model format")
            return

        print("📰 Fake News Detection CLI")
        print("=" * 40)

        while True:
            print("\nEnter news text (or 'quit' to exit):")
            text = input("> ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                break

            if not text:
                continue

            prediction, confidence = predict_news(text, model_package)

            print(f"\n🔍 Prediction: {prediction} News")
            print(f"📊 Confidence: {confidence}%")

            if confidence < 60:
                print("⚠️  Low confidence - results may be unreliable")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()