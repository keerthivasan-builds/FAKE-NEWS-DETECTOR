# Fake News Detection Web Application

A machine learning-powered web application that detects fake news using Natural Language Processing and Logistic Regression.

## Features

- **News Analysis**: Enter news headlines or articles to check authenticity
- **Machine Learning Model**: Trained Logistic Regression model with TF-IDF vectorization
- **Confidence Score**: Displays prediction confidence percentage
- **History Tracking**: Saves all checked news in a database
- **Statistics Dashboard**: View basic statistics of checked news
- **Clean UI**: Simple and responsive web interface

## Installation

1. **Clone or download the project** to your local machine.

2. **Install Python dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. *(Optional but recommended)* **Import a real dataset (recommended for best accuracy)**:
   - Download a dataset such as `fake_and_real_news.csv` (common on Kaggle)
   - Run:
     ```
     python import_dataset.py --input path/to/fake_and_real_news.csv
     ```

4. *(Optional but useful for small demos)* **Generate a larger synthetic dataset**:
   ```
   python augment_dataset.py
   ```

5. *(Optional)* **Filter the dataset for India-related news**:
   ```
   python filter_india_news.py
   ```
   This will create `dataset/fake_news_dataset_india.csv` using simple keyword matching.

6. *(Optional)* **Download a public fake news dataset**:
   ```
   python download_dataset.py --url https://<your-dataset-url>.csv
   # or (Kaggle):
   python download_dataset.py --kaggle cliveth/fake-news-detection
   ```

7. **Train the model** (re-trains based on the latest dataset):
   ```
   python model.py
   ```

6. **Run the application**:
   ```
   python app.py
   ```

7. *(Optional for live Google search verification)* **Configure Google API in `.env`**:
   - Create a `.env` file in the project root.
   - Add:
     ```
     GOOGLE_API_KEY=your_google_api_key
     GOOGLE_CSE_ID=your_custom_search_engine_id
     ```
   - Restart the app after updating `.env`.

5. **Open your browser** and go to `http://127.0.0.1:5000/`

## Project Structure

```
fake-news-detector/
├── app.py                 # Main Flask application
├── model.py               # Machine learning training script
├── requirements.txt       # Python dependencies
├── fake_news_model.pkl    # Trained model (generated)
├── tfidf_vectorizer.pkl   # TF-IDF vectorizer (generated)
├── fake_news.db          # SQLite database (generated)
├── dataset/
│   └── fake_news_dataset.csv  # Training dataset
├── templates/
│   ├── index.html         # Main page
│   ├── result.html        # Results page
│   ├── history.html       # History page
│   └── stats.html         # Statistics page
└── static/
    └── style.css          # CSS styles
```

## How to Run in VS Code

1. **Open the project folder** in VS Code:
   - File > Open Folder
   - Select the `fake-news-detector` folder

2. **Install dependencies**:
   - Open terminal in VS Code (Ctrl + `)
   - Run: `pip install -r requirements.txt`

3. **Run the training script** (if needed):
   - In terminal: `python model.py`

4. **Run the Flask app**:
   - In terminal: `python app.py`
   - Or use VS Code's run button if configured

5. **Access the application**:
   - Open browser to `http://127.0.0.1:5000/`

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, Logistic Regression, TF-IDF
- **Database**: SQLite with SQLAlchemy
- **Frontend**: HTML, CSS
- **Data Processing**: pandas, numpy

## Model Training

The model is trained on a dataset of news articles labeled as real (1) or fake (0). The training process:

1. Combines title and text for better features
2. Uses TF-IDF vectorization to convert text to numerical features
3. Trains a Logistic Regression classifier
4. Saves the model and vectorizer for later use

## API Endpoints

- `/` - Main page with news input form
- `/check` - POST endpoint for news analysis
- `/history` - View checked news history
- `/stats` - View statistics dashboard

## License

This project is for educational purposes.