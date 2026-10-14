#!/usr/bin/env python
# Test the new verification system

from app import predict_news, build_search_query, google_custom_search, analyze_search_verification

test_cases = [
    "Coronavirus disease 2019 (COVID-19) is a contagious disease caused by the coronavirus SARS-CoV-2.",
    "The government approved a new vaccine policy yesterday.",
    "Elvis Presley is still alive and performing secret concerts.",
    "Climate change is real and caused by human activities.",
]

print("=" * 80)
print("FAKE NEWS DETECTION - FULL TEXT ANALYSIS WITH API VERIFICATION")
print("=" * 80)

for text in test_cases:
    print(f"\n[INPUT] Text: {text[:70]}...")
    
    # ML Model Prediction
    prediction, confidence = predict_news(text)
    print(f"[ML MODEL] {prediction} ({confidence}%)")
    
    # Build search query using MORE of the text (not just first 16 words)
    search_query = build_search_query(text)
    print(f"[SEARCH QUERY] {search_query}")
    
    # Search online for verification
    search_results = google_custom_search(search_query, max_results=3)
    sources_count = len(search_results.get('sources', []))
    print(f"[SOURCES FOUND] {sources_count}")
    
    # Analyze verification from search results
    verification = analyze_search_verification(text, search_results)
    print(f"[VERIFICATION] {verification['explanation']}")
    print(f"[CONFIDENCE] {verification['confidence']}%")
    print(f"[VERIFIED] {verification['verified']}")
    
    # Combined Verdict
    print(f"[FINAL VERDICT]", end=" ")
    if verification['verified'] is True:
        print("CLAIM IS VERIFIED by credible sources")
    elif verification['verified'] is False:
        print("CLAIM NOT VERIFIED")
    else:
        print("UNVERIFIABLE - No sources found or inconclusive")
        
    print("-" * 80)
