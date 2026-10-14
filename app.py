# app.py - Main Backend File for Fake News Detection Web App
# pyright: basic

from __future__ import annotations

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

import pickle
import os
import re
import requests

try:
    from deep_translator import GoogleTranslator  # type: ignore[reportMissingImports]
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

import numpy as np  # type: ignore
from scipy.sparse import hstack, csr_matrix  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fake_news.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

SUPPORTED_INPUT_LANGUAGES = {
    'auto': {'label': 'Auto Detect', 'code': 'auto'},
    'en': {'label': 'English', 'code': 'en'},
    'ta': {'label': 'Tamil', 'code': 'ta'},
    'hi': {'label': 'Hindi', 'code': 'hi'},
}


def load_local_env(env_path='.env'):
    """Load KEY=VALUE pairs from a local .env file into os.environ."""
    if not os.path.exists(env_path):
        return

    with open(env_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


load_local_env()

# Optional Google Custom Search credentials for live web verification.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '').strip()
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID', '').strip()

# ── Load the trained model ─────────────────────────────────────────────────────
model_path = 'fake_news_model.pkl'
dataset_path = os.path.join('dataset', 'fake_news_dataset.csv')

model_package = None   # new ensemble dict
legacy_model = None    # old Pipeline model (fallback)
vectorizer = None      # TF-IDF vectorizer

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict) and loaded.get('type') == 'ensemble_v4':
        model_package = loaded
        vectorizer = loaded['tfidf_word']
    elif isinstance(loaded, dict) and loaded.get('type') == 'ensemble_v3':
        model_package = loaded
        vectorizer = loaded['tfidf']
    elif isinstance(loaded, dict) and loaded.get('type') == 'ensemble_v2':
        model_package = loaded
        vectorizer = loaded['tfidf']
    else:
        # Legacy pipeline model
        legacy_model = loaded
        if hasattr(legacy_model, 'named_steps') and 'tfidf' in legacy_model.named_steps:  # type: ignore
            vectorizer = legacy_model.named_steps['tfidf']  # type: ignore


# ── Stylometric feature extractor (mirrors model.py) ──────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_stylometric_features(texts):
    features = []
    for text in texts:
        text = str(text)
        length = len(text) + 1
        num_words = len(text.split()) + 1
        num_sentences = len([s for s in text.split('.') if s.strip()]) + 1

        exclamation_ratio = text.count('!') / length
        question_ratio = text.count('?') / length
        uppercase_ratio = sum(1 for c in text if c.isupper()) / length
        avg_word_len = sum(len(w) for w in text.split()) / num_words
        punct_density = sum(1 for c in text if c in '.,!?;:') / length

        words = text.split()
        short_words = sum(1 for w in words if len(w) <= 3) / num_words
        long_words = sum(1 for w in words if len(w) >= 8) / num_words
        avg_sentence_len = num_words / num_sentences

        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1) / num_words
        title_case_words = sum(1 for w in words if w.istitle()) / num_words
        unique_punct = len(set(c for c in text if c in '.,!?;:()[]{}')) / length

        features.append([
            min(length / 5000, 1.0),
            exclamation_ratio * 100,
            question_ratio * 100,
            uppercase_ratio * 100,
            avg_word_len / 10.0,
            punct_density * 100,
            short_words,
            long_words,
            min(avg_sentence_len / 50, 1.0),
            all_caps_words,
            title_case_words,
            unique_punct * 100,
        ])
    return np.array(features, dtype=np.float32)


# ── Load dataset for similarity examples ──────────────────────────────────────
similarity_df = None
similarity_matrix = None
if os.path.exists(dataset_path) and vectorizer is not None:
    try:
        similarity_df = pd.read_csv(dataset_path)
        similarity_df['content'] = (
            similarity_df.get('title', '').fillna('') + ' ' + similarity_df.get('text', '').fillna('')  # type: ignore
        )
        similarity_matrix = vectorizer.transform(similarity_df['content'].tolist())
    except Exception:
        similarity_df = None
        similarity_matrix = None


# ── Database Model ─────────────────────────────────────────────────────────────
class NewsCheck(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()


# ── Model configuration ───────────────────────────────────────────────────


def predict_news(news_text):
    """
    Run prediction using the best available model.
    Applies smart heuristic bias corrections for:
      - Sensationalism (fake news signal)
      - Credible reporting style (real news signal)
      - Excessive capitalisation (clickbait signal)
      - Very short text (low confidence)
    Returns (prediction, confidence_percent)
    """
    if model_package:
        threshold = float(model_package.get('decision_threshold', 0.5))
        threshold = max(0.48, min(threshold, 0.65))
        weight_cfg = model_package.get('weights', {})

        cleaned = clean_text(news_text)

        if model_package.get('type') == 'ensemble_v4':
            tfidf_word = model_package['tfidf_word']
            tfidf_char = model_package['tfidf_char']
            models = model_package.get('models', {})
            active_tfidf = tfidf_word

            word_feat = tfidf_word.transform([cleaned])
            char_feat = tfidf_char.transform([cleaned])
            style_feat = csr_matrix(extract_stylometric_features([cleaned]))
            X = hstack([word_feat, char_feat, style_feat])

            model_probs = []
            for name in ['pac', 'lr', 'svc']:
                model_obj = models.get(name)
                if model_obj is not None:
                    model_probs.append((name, model_obj.predict_proba(X)[0]))
        else:
            tfidf = model_package['tfidf']
            pac = model_package['pac']
            lr = model_package['lr']
            rf = model_package.get('rf')  # May not exist in older models
            active_tfidf = tfidf

            tfidf_feat = tfidf.transform([cleaned])
            style_feat = csr_matrix(extract_stylometric_features([cleaned]))
            X = hstack([tfidf_feat, style_feat])

            pac_proba = pac.predict_proba(X)[0]
            lr_proba = lr.predict_proba(X)[0]
            model_probs = [('pac', pac_proba), ('lr', lr_proba)]

            if rf is not None:
                rf_proba = rf.predict_proba(X)[0]
                model_probs.append(('rf', rf_proba))

        raw_weights = []
        for name, _ in model_probs:
            w = float(weight_cfg.get(name, 1.0))
            raw_weights.append(max(w, 0.0))

        if sum(raw_weights) <= 0:
            raw_weights = [1.0] * len(model_probs)

        total_weight = float(sum(raw_weights))
        norm_weights = [w / total_weight for w in raw_weights]

        ensemble_proba = np.zeros(2, dtype=np.float64)
        for idx, (_, proba) in enumerate(model_probs):
            ensemble_proba += norm_weights[idx] * proba

        real_prob = float(ensemble_proba[1])
        model_real_probs = np.array([float(p[1]) for _, p in model_probs], dtype=np.float64)
        consensus_std = float(np.std(model_real_probs))

        # Very short text and low model agreement -> reduce overconfidence.
        word_count = len(cleaned.split())
        
        # Only apply shrinkage to very short text (< 15 words)
        # Longer legitimate text should use raw probabilities
        if word_count < 15:
            length_factor = max(0.0, min((word_count - 3) / 12.0, 1.0))
            shrink = 0.60 + (0.40 * length_factor)
            real_prob = 0.5 + (real_prob - 0.5) * shrink

        if consensus_std > 0.18:
            real_prob = 0.5 + (real_prob - 0.5) * 0.75

        # Out-of-distribution guardrail:
        # Only apply strict penalty to very short text (< 15 words) with low vocab.
        # Longer legitimate text should not be penalized for technical terms.
        tokens = re.findall(r'[a-z]+', cleaned)
        if tokens and len(tokens) < 15:
            vocab = getattr(active_tfidf, 'vocabulary_', {})
            known = sum(1 for tok in tokens if tok in vocab)
            known_ratio = known / len(tokens)

            if known_ratio < 0.50:
                real_prob = min(real_prob, 0.48)

        # Authority/health keyword boost:
        # Medical/scientific text from recognized sources should get a small boost
        authority_keywords = ['who', 'cdc', 'covid', 'coronavirus', 'vaccine', 'health', 'disease', 'virus', 'pandemic', 'government', 'ministry', 'approved', 'confirmed']
        found_keywords = sum(1 for kw in authority_keywords if kw in cleaned)
        
        if found_keywords >= 2 and word_count >= 10:
            real_prob = min(0.55, real_prob + 0.05)

        real_prob = float(np.clip(real_prob, 0.01, 0.99))

        # ── Final classification ──────────────────────────────────────────────
        prediction = 'Real' if real_prob >= threshold else 'Fake'

        denom = max(threshold, 1.0 - threshold)
        confidence_distance = abs(real_prob - threshold) / denom
        confidence = int(round(50 + (confidence_distance * 50)))

        if consensus_std < 0.08 and word_count >= 40:
            confidence += 3
        if consensus_std > 0.20:
            confidence -= 5

        confidence = int(np.clip(confidence, 50, 99))
        return prediction, confidence

    if legacy_model:
        if hasattr(legacy_model, 'named_steps'):
            model_input = [news_text]
        elif vectorizer:
            model_input = vectorizer.transform([news_text])
        else:
            return None, 0

        proba = legacy_model.predict_proba(model_input)[0]  # type: ignore
        prediction = 'Real' if proba[1] > 0.5 else 'Fake'
        confidence = int(round(max(proba) * 100))
        confidence = min(confidence, 99)
        return prediction, confidence

    return None, 0


def build_search_query(news_text, max_words=None):
    """Build search query using more of the input text for better API verification."""
    cleaned = re.sub(r'\s+', ' ', str(news_text)).strip()
    words = cleaned.split()
    
    # Use 60% of text or minimum 25 words, maximum 50 words for better context
    if max_words is None:
        max_words = max(25, min(len(words), 50))
    
    return ' '.join(words[:max_words])


def preprocess_input_text(news_text, input_language='auto'):
    """
    Normalize incoming text for model prediction.
    Translates Tamil/Hindi (or auto-detected language) to English when possible.
    Returns (processed_text, translation_meta).
    """
    lang = str(input_language or 'auto').strip().lower()
    if lang not in SUPPORTED_INPUT_LANGUAGES:
        lang = 'auto'

    cleaned = str(news_text or '').strip()
    meta = {
        'selected_language': lang,
        'selected_language_label': SUPPORTED_INPUT_LANGUAGES[lang]['label'],
        'translated': False,
        'translator_available': TRANSLATOR_AVAILABLE,
        'status': 'No translation needed.',
    }

    if not cleaned:
        return cleaned, meta

    if lang == 'en':
        return cleaned, meta

    if not TRANSLATOR_AVAILABLE:
        meta['status'] = 'Translator is not installed; using original text.'
        return cleaned, meta

    try:
        source_lang = SUPPORTED_INPUT_LANGUAGES[lang]['code']
        translated_text = GoogleTranslator(source=source_lang, target='en').translate(cleaned)
        translated_text = str(translated_text or '').strip()

        if translated_text:
            meta['translated'] = True
            meta['status'] = 'Input translated to English for model prediction.'
            return translated_text, meta

        meta['status'] = 'Translation returned empty text; using original input.'
        return cleaned, meta
    except Exception:
        meta['status'] = 'Translation failed; using original input.'
        return cleaned, meta


def analyze_search_verification(news_text, search_results):
    """
    Analyze search results to verify if the news claim is supported by credible sources.
    Returns: {verified: bool, confidence: float (0-100), sources_found: int, explanation: str}
    """
    sources = search_results.get('sources', [])
    
    if not sources:
        return {
            'verified': None,
            'confidence': 0,
            'sources_found': 0,
            'explanation': 'No search results found - cannot verify claim'
        }
    
    cleaned = clean_text(news_text)
    keywords = ['confirm', 'verified', 'fact', 'source', 'official', 'report', 'news', 'statement', 'said', 'report']
    
    # Credible source indicators
    credible_domains = ['bbc', 'reuters', 'ap', 'apnews', 'cnn', 'aljazeera', 'guardian', 'nytimes', 
                        'washington', 'bbc', 'who', 'health', 'gov', 'edu', 'reuters', 'official']
    
    verification_score = 0
    supporting_sources = 0
    
    for source in sources:
        title = source.get('title', '').lower()
        snippet = source.get('snippet', '').lower()
        link = source.get('link', '').lower()
        
        # Check if source is credible
        is_credible = any(domain in link for domain in credible_domains)
        
        # Check if source mentions key terms from the news text
        text_words = set(cleaned.split())
        source_text = f"{title} {snippet}"
        matches = sum(1 for word in text_words if len(word) > 3 and word in source_text)
        
        if matches > 0 or is_credible:
            supporting_sources += 1
            verification_score += 20 if is_credible else 10
    
    final_confidence = min(supporting_sources * 30, 90)  # Max 90% from search results
    is_verified = supporting_sources >= 2 or (supporting_sources == 1 and verification_score >= 25)
    
    explanation = ""
    if supporting_sources >= 2:
        explanation = f"Found {supporting_sources} credible sources supporting this claim"
    elif supporting_sources == 1:
        explanation = "Found 1 source that mentions this claim"
    else:
        explanation = "Search results do not corroborate this specific claim"
    
    return {
        'verified': is_verified,
        'confidence': int(final_confidence),
        'sources_found': supporting_sources,
        'explanation': explanation
    }



def google_custom_search(query, max_results=3):
    """
    Search Google Custom Search JSON API with DuckDuckGo fallback.
    Returns dict with keys: ok, status, sources.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return ddgs_search(query, max_results)

    try:
        response = requests.get(
            'https://www.googleapis.com/customsearch/v1',
            params={
                'key': GOOGLE_API_KEY,
                'cx': GOOGLE_CSE_ID,
                'q': query,
                'num': max(1, min(int(max_results), 10)),
            },
            timeout=10,
        )

        if response.status_code == 200:
            payload = response.json()
            items = payload.get('items', [])
            sources = []
            for item in items:
                sources.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                })

            if sources:
                return {'ok': True, 'status': 'Search via Google API.', 'sources': sources}
            else:
                return ddgs_search(query, max_results)

        if response.status_code == 403:
            return ddgs_search(query, max_results)

        return ddgs_search(query, max_results)
    except Exception as e:
        return ddgs_search(query, max_results)


def ddgs_search(query, max_results=3):
    """
    Search using DuckDuckGo (free, no API key required).
    Returns dict with keys: ok, status, sources.
    """
    if not DDGS_AVAILABLE:
        return {
            'ok': False,
            'status': 'No search backend available (DuckDuckGo not installed).',
            'sources': [],
        }

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        sources = []
        for result in results:
            sources.append({
                'title': result.get('title', ''),
                'link': result.get('href', ''),
                'snippet': result.get('body', ''),
            })

        if sources:
            return {'ok': True, 'status': 'Search via DuckDuckGo.', 'sources': sources}
        else:
            return {'ok': True, 'status': 'No sources found.', 'sources': []}
    except Exception as e:
        return {'ok': False, 'status': f'Search failed: {str(e)[:100]}', 'sources': []}


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', supported_languages=SUPPORTED_INPUT_LANGUAGES)


@app.route('/check', methods=['POST'])
def check_news():
    """Process text input, predict fake/real, and show result."""
    if not model_package and not legacy_model:
        return "Model not loaded. Please train the model first."

    news_text = request.form.get('news_text', '').strip()
    input_language = request.form.get('input_language', 'auto').strip().lower()
    if not news_text:
        return render_template('result.html', error="Please enter so me news text.")

    processed_text, translation_meta = preprocess_input_text(news_text, input_language)

    # Check for exact dataset match first
    input_lower = processed_text.lower()
    exact_match = False
    prediction = None
    confidence = 0

    if similarity_df is not None:
        for col in ['title', 'text']:
            if col in similarity_df.columns:
                matches = similarity_df[
                    similarity_df[col].astype(str).str.strip().str.lower() == input_lower
                ]
                if not matches.empty:
                    exact_match = True
                    label = matches.iloc[0]['label']
                    prediction = 'Real' if label == 1 else 'Fake'
                    confidence = 95
                    break

    if not exact_match:
        prediction, confidence = predict_news(processed_text)
        if prediction is None:
            return "Model not configured correctly. Please retrain."

    # Find similar examples
    similar_real, similar_fake = [], []
    if similarity_df is not None and similarity_matrix is not None:
        try:
            query_vec = vectorizer.transform([processed_text])  # type: ignore
            scores = cosine_similarity(query_vec, similarity_matrix)[0]
            similarity_df['score'] = scores
            top = similarity_df.sort_values('score', ascending=False).head(10)
            similar_real = top[top['label'] == 1].head(3).to_dict('records')  # type: ignore
            similar_fake = top[top['label'] == 0].head(3).to_dict('records')  # type: ignore
        except Exception:
            pass

    # Optional live web verification via Google API.
    search_query = build_search_query(processed_text)
    web_result = google_custom_search(search_query, max_results=3)
    
    # Analyze verification results
    verification = analyze_search_verification(processed_text, web_result)

    # Save to DB
    db.session.add(NewsCheck(content=news_text, prediction=prediction, confidence=confidence))  # type: ignore[call-arg]
    db.session.commit()

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=f"{confidence:.2f}",
        news_text=news_text,
        similar_real=similar_real,
        similar_fake=similar_fake,
        google_status=web_result.get('status', ''),
        google_sources=web_result.get('sources', []),
        verification=verification,
        search_query=search_query,
        input_language_label=translation_meta['selected_language_label'],
        translation_status=translation_meta['status'],
        translated_text=processed_text if translation_meta['translated'] else '',
    )


@app.route('/api/google-check', methods=['POST'])
def api_google_check():
    """JSON API: prediction + optional Google Custom Search evidence + verification."""
    data = request.get_json(silent=True) or {}
    news_text = str(data.get('news_text', '')).strip()
    input_language = str(data.get('input_language', 'auto')).strip().lower()

    if not news_text:
        return jsonify({'ok': False, 'error': 'news_text is required'}), 400

    if not model_package and not legacy_model:
        return jsonify({'ok': False, 'error': 'Model not loaded. Train the model first.'}), 500

    processed_text, translation_meta = preprocess_input_text(news_text, input_language)

    prediction, confidence = predict_news(processed_text)
    if prediction is None:
        return jsonify({'ok': False, 'error': 'Model prediction failed.'}), 500

    search_query = build_search_query(processed_text)
    web_result = google_custom_search(search_query, max_results=3)
    
    # Analyze verification
    verification = analyze_search_verification(processed_text, web_result)

    return jsonify({
        'ok': True,
        'prediction': prediction,
        'confidence': confidence,
        'input_language': translation_meta['selected_language'],
        'input_language_label': translation_meta['selected_language_label'],
        'translation_status': translation_meta['status'],
        'translated_text': processed_text if translation_meta['translated'] else '',
        'search_query': search_query,
        'google': web_result,
        'verification': verification,
    })


@app.route('/history')
def history():
    checks = NewsCheck.query.order_by(NewsCheck.timestamp.desc()).all()
    return render_template('history.html', checks=checks)


@app.route('/stats')
def stats():
    total_checks = NewsCheck.query.count()
    real_count = NewsCheck.query.filter_by(prediction='Real').count()
    fake_count = NewsCheck.query.filter_by(prediction='Fake').count()
    real_pct = (real_count / total_checks * 100) if total_checks > 0 else 0
    fake_pct = (fake_count / total_checks * 100) if total_checks > 0 else 0
    return render_template('stats.html',
                           total=total_checks,
                           real=real_count,
                           fake=fake_count,
                           real_pct=f"{real_pct:.1f}",
                           fake_pct=f"{fake_pct:.1f}")


if __name__ == '__main__':
    app.run(debug=True, port=5001)