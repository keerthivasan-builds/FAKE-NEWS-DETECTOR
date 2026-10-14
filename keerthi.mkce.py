# model.py - Machine Learning Training Script for Fake News Detection
# Uses a high-accuracy ensemble: TF-IDF + PassiveAggressiveClassifier + LogisticRegression

import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text):
    """Enhanced text cleaning for better feature extraction."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)        # Remove URLs
    text = re.sub(r'@\w+', '', text)                    # Remove mentions
    text = re.sub(r'#\w+', '', text)                    # Remove hashtags
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)   # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_stylometric_features(texts):
    """
    Extract comprehensive writing-style features that distinguish fake news:
    - Text length and structure
    - Punctuation patterns (fake news uses more ! and ?)
    - Capitalization patterns (fake news uses MORE CAPS)
    - Word complexity metrics
    - Readability scores
    - Sentiment indicators
    """
    features = []
    for text in texts:
        text = str(text)
        length = len(text) + 1  # avoid zero division
        num_words = len(text.split()) + 1
        num_sentences = len([s for s in text.split('.') if s.strip()]) + 1

        # Basic metrics
        exclamation_ratio = text.count('!') / length
        question_ratio = text.count('?') / length
        uppercase_ratio = sum(1 for c in text if c.isupper()) / length
        avg_word_len = sum(len(w) for w in text.split()) / num_words
        punct_density = sum(1 for c in text if c in '.,!?;:') / length

        # Advanced metrics
        words = text.split()
        short_words = sum(1 for w in words if len(w) <= 3) / num_words
        long_words = sum(1 for w in words if len(w) >= 8) / num_words
        avg_sentence_len = num_words / num_sentences

        # Capitalization patterns
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1) / num_words
        title_case_words = sum(1 for w in words if w.istitle()) / num_words

        # Punctuation variety
        unique_punct = len(set(c for c in text if c in '.,!?;:()[]{}')) / length

        features.append([
            min(length / 5000, 1.0),   # normalized length
            exclamation_ratio * 100,
            question_ratio * 100,
            uppercase_ratio * 100,
            avg_word_len / 10.0,
            punct_density * 100,
            short_words,
            long_words,
            min(avg_sentence_len / 50, 1.0),  # normalized sentence length
            all_caps_words,
            title_case_words,
            unique_punct * 100,
        ])
    return np.array(features, dtype=np.float32)


def train_model():
    """
    Train an improved fake news detection model.
    
    Strategy:
    - TF-IDF with title + body (1-3 grams, 15k features)
    - Stylometric features (writing style signals)
    - PassiveAggressiveClassifier (best for text classification, ~95% accuracy)
    - LogisticRegression as secondary for calibrated probabilities
    - Soft-voting ensemble of PAC + LR
    """

    # Prefer augmented dataset (267MB) over base dataset
    aug_path = os.path.join('dataset', 'fake_news_dataset_augmented.csv')
    base_path = os.path.join('dataset', 'fake_news_dataset.csv')
    dataset_path = aug_path if os.path.exists(aug_path) else base_path

    print(f"Loading dataset from: {dataset_path}")

    # For faster testing, use a sample of the data
    if os.path.exists(aug_path):
        # Load only 5,000 samples for a quick test
        df = pd.read_csv(dataset_path, nrows=5000)
        print(f"Using sample of {len(df)} rows for quick test")
    else:
        df = pd.read_csv(dataset_path)
        print(f"Total samples: {len(df)}")

    # Prepare text
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + ' ' + df['text']
    df['content_clean'] = df['content'].apply(clean_text)

    X = df['content_clean']
    y = df['label']

    print(f"Class balance:\n{y.value_counts()}\n")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ── Build TF-IDF features ──────────────────────────────────────────────
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),       # unigrams and bigrams only (faster)
        max_df=0.9,
        min_df=5,
        max_features=5000,        # Reduced from 15k for faster training
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # ── Build stylometric features ─────────────────────────────────────────
    print("Extracting stylometric features...")
    X_train_style = csr_matrix(extract_stylometric_features(X_train.tolist()))
    X_test_style = csr_matrix(extract_stylometric_features(X_test.tolist()))

    # Combine TF-IDF + stylometric features
    X_train_combined = hstack([X_train_tfidf, X_train_style])
    X_test_combined = hstack([X_test_tfidf, X_test_style])

    # ── Train models ───────────────────────────────────────────────────────

    print("Training PassiveAggressiveClassifier...")
    pac = CalibratedClassifierCV(
        PassiveAggressiveClassifier(C=0.5, max_iter=1000, random_state=42),
        cv=3
    )
    pac.fit(X_train_combined, y_train)

    print("Training LogisticRegression...")
    lr = LogisticRegression(C=2.0, max_iter=1000, random_state=42, solver='lbfgs')
    lr.fit(X_train_combined, y_train)

    print("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=50,          # Reduced from 100
        max_depth=15,             # Reduced from 20
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_combined, y_train)

    # Evaluate individual models
    pac_pred = pac.predict(X_test_combined)
    lr_pred = lr.predict(X_test_combined)
    rf_pred = rf.predict(X_test_combined)

    print(f"\nPAC Test accuracy:  {accuracy_score(y_test, pac_pred):.4f}")
    print(f"LR  Test accuracy:  {accuracy_score(y_test, lr_pred):.4f}")
    print(f"RF  Test accuracy:  {accuracy_score(y_test, rf_pred):.4f}")

    # Skip cross-validation for faster training
    print("\nSkipping cross-validation for faster training...")

    # Get probabilities for soft voting ensemble
    pac_proba = pac.predict_proba(X_test_combined)
    lr_proba = lr.predict_proba(X_test_combined)
    rf_proba = rf.predict_proba(X_test_combined)
    ensemble_proba = (pac_proba + lr_proba + rf_proba) / 3

    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble accuracy:  {ensemble_acc:.4f}")
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test, ensemble_pred, zero_division=0))

    # ── Save model package ─────────────────────────────────────────────────
    model_package = {
        'tfidf': tfidf,
        'pac': pac,
        'lr': lr,
        'rf': rf,
        'type': 'ensemble_v3',
    }
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)

    print("\n[OK] Model saved to fake_news_model.pkl")
    print(f"[OK] Final ensemble accuracy: {ensemble_acc*100:.2f}%")


if __name__ == '__main__':
    train_model()
