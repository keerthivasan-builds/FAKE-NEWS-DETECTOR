import pandas as pd  # type: ignore

def add_shivalik():
    df = pd.read_csv('dataset/fake_news_dataset.csv')
    new_rows = pd.DataFrame([
        {'title': 'LPG tanker Shivalik reaches Gujarat', 'text': 'The LPG tanker Shivalik successfully reached Gujarat\'s Mundra Port after a safe transit through the conflict-hit Strait of Hormuz, bringing much-needed cooking gas supplies.', 'label': 1},
        {'title': 'Shivalik safely docks at Mundra', 'text': 'The LPG tanker Shivalik successfully reached Gujarat\'s Mundra Port after a safe transit through the conflict-hit Strait of Hormuz, bringing much-needed cooking gas supplies.', 'label': 1},
        {'title': 'Cooking gas supplies arrive', 'text': 'The LPG tanker Shivalik successfully reached Gujarat\'s Mundra Port after a safe transit through the conflict-hit Strait of Hormuz, bringing much-needed cooking gas supplies.', 'label': 1}
    ])
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv('dataset/fake_news_dataset.csv', index=False)
    print("Added Shivalik examples to dataset")

if __name__ == '__main__':
    add_shivalik()
