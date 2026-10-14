from duckduckgo_search import DDGS
import sys

# Test 1: Full text
news_text = "The LPG tanker 'Shivalik' successfully reached Gujarat's Mundra Port after a safe transit through the conflict-hit Strait of Hormuz, bringing much-needed cooking gas supplies."

print("Testing full sentence search:")
with DDGS() as ddgs:
    results = list(ddgs.text(news_text, max_results=3))
    print(f"Got {len(results)} results")
    if results: print("-", results[0].get('title'))

# Test 2: Just 10 words
first_10 = ' '.join(news_text.split()[:10])
print(f"\nTesting 10 words: '{first_10}'")
with DDGS() as ddgs:
    results = list(ddgs.text(first_10, max_results=3))
    print(f"Got {len(results)} results")
    if results: print("-", results[0].get('title'))
