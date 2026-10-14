import requests


def test_app():
    # Test the real unseen news snippet
    text_content = "The LPG tanker 'Shivalik' successfully reached Gujarat's Mundra Port after a safe transit through the conflict-hit Strait of Hormuz, bringing much-needed cooking gas supplies."
    response = requests.post(
        'http://127.0.0.1:5001/api/google-check',
        json={'news_text': text_content},
        timeout=20,
    )
    print("Test Unseen Real News:")
    if response.ok:
        payload = response.json()
        print(f"Prediction: {payload.get('prediction')} ({payload.get('confidence')}%)")
        google = payload.get('google', {})
        print(f"Google status: {google.get('status')}")
        print(f"Google sources: {len(google.get('sources', []))}")
    else:
        print(f"Request failed: {response.status_code}")
        print(response.text)

    # Test an obvious fake news hoax
    fake_content = "Bill Gates announces he is buying Apple for 500 billion dollars and shutting down the iPhone immediately, shocking fact check."
    response = requests.post(
        'http://127.0.0.1:5001/api/google-check',
        json={'news_text': fake_content},
        timeout=20,
    )
    print("\nTest Fake Target text:")
    if response.ok:
        payload = response.json()
        print(f"Prediction: {payload.get('prediction')} ({payload.get('confidence')}%)")
        google = payload.get('google', {})
        print(f"Google status: {google.get('status')}")
        print(f"Google sources: {len(google.get('sources', []))}")
    else:
        print(f"Request failed: {response.status_code}")
        print(response.text)


if __name__ == '__main__':
    test_app()
