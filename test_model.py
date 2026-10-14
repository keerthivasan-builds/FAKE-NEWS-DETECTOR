#!/usr/bin/env python3
"""
Simple test script for the fake news detection model
"""

import pickle
import os

def test_model():
    # Load the model
    model_path = 'fake_news_model.pkl'
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please run 'python model.py' first.")
        return

    try:
        with open(model_path, 'rb') as f:
            loaded = pickle.load(f)

        print("✅ Model loaded successfully!")

        if isinstance(loaded, dict):
            print(f"Model type: {loaded.get('type', 'unknown')}")
            print(f"Model components: {list(loaded.keys())}")

            # Test with sample text
            test_texts = [
                "Breaking: Scientists discover new planet with life!",
                "Local man wins lottery for the third time this year",
                "Government announces new tax policy affecting millions"
            ]

            for text in test_texts:
                print(f"\nTesting: '{text[:50]}...'")
                # Simple prediction (would need full app.py logic)
                print("Model ready for prediction!")

        else:
            print("Legacy model format detected")

    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == '__main__':
    test_model()