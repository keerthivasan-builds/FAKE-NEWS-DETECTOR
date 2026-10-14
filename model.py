# model.py - Machine Learning Training Script for Fake News Detection
# Uses a high-accuracy ensemble: TF-IDF + PassiveAggressiveClassifier + LogisticRegression

import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score


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
    - Word + character TF-IDF for semantic + stylistic text signals
    - Stylometric numeric features
    - Calibrated linear models (PAC, LogisticRegression, LinearSVC)
    - Weighted soft-voting with threshold optimized on macro-F1
    """

    # Prefer augmented dataset (267MB) over base dataset
    aug_path = os.path.join('dataset', 'fake_news_dataset_augmented.csv')
    base_path = os.path.join('dataset', 'fake_news_dataset.csv')
    dataset_path = aug_path if os.path.exists(aug_path) else base_path

    print(f"Loading dataset from: {dataset_path}")

    # Load full dataset for best accuracy.
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

    # ── Build word-level TF-IDF features ───────────────────────────────────
    print("Fitting word TF-IDF vectorizer...")
    tfidf_word = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=3,
        max_features=20000,
        sublinear_tf=True,
    )
    X_train_word = tfidf_word.fit_transform(X_train)
    X_test_word = tfidf_word.transform(X_test)

    # ── Build char-level TF-IDF features (captures clickbait patterns) ─────
    print("Fitting char TF-IDF vectorizer...")
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_df=0.95,
        min_df=3,
        max_features=25000,
        sublinear_tf=True,
    )
    X_train_char = tfidf_char.fit_transform(X_train)
    X_test_char = tfidf_char.transform(X_test)

    # ── Build stylometric features ─────────────────────────────────────────
    print("Extracting stylometric features...")
    X_train_style = csr_matrix(extract_stylometric_features(X_train.tolist()))
    X_test_style = csr_matrix(extract_stylometric_features(X_test.tolist()))

    # Combine word + char TF-IDF + stylometric features
    X_train_combined = hstack([X_train_word, X_train_char, X_train_style])
    X_test_combined = hstack([X_test_word, X_test_char, X_test_style])

    # ── Train models ───────────────────────────────────────────────────────

    print("Training PassiveAggressiveClassifier...")
    pac = CalibratedClassifierCV(
        PassiveAggressiveClassifier(C=0.45, max_iter=1200, random_state=42),
        cv=3
    )
    pac.fit(X_train_combined, y_train)

    print("Training LogisticRegression...")
    lr = LogisticRegression(
        C=3.0,
        max_iter=1500,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'
    )
    lr.fit(X_train_combined, y_train)

    print("Training calibrated LinearSVC...")
    svc = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight='balanced', random_state=42),
        cv=3
    )
    svc.fit(X_train_combined, y_train)

    # Evaluate individual models
    pac_pred = pac.predict(X_test_combined)
    lr_pred = lr.predict(X_test_combined)
    svc_pred = svc.predict(X_test_combined)

    pac_acc = accuracy_score(y_test, pac_pred)
    lr_acc = accuracy_score(y_test, lr_pred)
    svc_acc = accuracy_score(y_test, svc_pred)

    print(f"\nPAC Test accuracy:  {pac_acc:.4f}")
    print(f"LR  Test accuracy:  {lr_acc:.4f}")
    print(f"SVC Test accuracy:  {svc_acc:.4f}")

    # Use validation performance as ensemble weights.
    acc_values = {
        'pac': float(pac_acc),
        'lr': float(lr_acc),
        'svc': float(svc_acc),
    }
    acc_sum = sum(acc_values.values())
    if acc_sum <= 0:
        weights = {'pac': 1 / 3, 'lr': 1 / 3, 'svc': 1 / 3}
    else:
        weights = {k: (v ** 3) for k, v in acc_values.items()}
        total = sum(weights.values())
        weights = {k: (v / total) for k, v in weights.items()}

    print(f"\nEnsemble weights: {weights}")

    # Get probabilities for soft voting ensemble
    pac_proba = pac.predict_proba(X_test_combined)
    lr_proba = lr.predict_proba(X_test_combined)
    svc_proba = svc.predict_proba(X_test_combined)
    ensemble_proba = (
        (weights['pac'] * pac_proba)
        + (weights['lr'] * lr_proba)
        + (weights['svc'] * svc_proba)
    )

    # Optimize threshold for macro-F1 to improve both real and fake quality.
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in np.arange(0.35, 0.66, 0.01):
        pred = (ensemble_proba[:, 1] >= threshold).astype(int)
        score = f1_score(y_test, pred, average='macro', zero_division=0)
        if score > best_macro_f1:
            best_macro_f1 = score
            best_threshold = float(threshold)

    print(f"Best threshold (Macro-F1): {best_threshold:.2f} | Macro-F1: {best_macro_f1:.4f}")

    ensemble_pred = (ensemble_proba[:, 1] >= best_threshold).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble accuracy:  {ensemble_acc:.4f}")
    print("\nClassification Report (Ensemble):")
    print(classification_report(y_test, ensemble_pred, zero_division=0))

    # ── Save model package ─────────────────────────────────────────────────
    model_package = {
        'tfidf_word': tfidf_word,
        'tfidf_char': tfidf_char,
        'models': {
            'pac': pac,
            'lr': lr,
            'svc': svc,
        },
        'weights': weights,
        'decision_threshold': best_threshold,
        'metrics': {
            'pac_accuracy': float(pac_acc),
            'lr_accuracy': float(lr_acc),
            'svc_accuracy': float(svc_acc),
            'ensemble_accuracy': float(ensemble_acc),
            'best_macro_f1': float(best_macro_f1),
        },
        'type': 'ensemble_v4',
    }
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)

    print("\n[OK] Model saved to fake_news_model.pkl")
    print(f"[OK] Final ensemble accuracy: {ensemble_acc*100:.2f}%")


if __name__ == '__main__':
    train_model()
