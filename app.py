from flask import Flask, render_template, request, jsonify
import re
import string
import requests
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import numpy as np

app = Flask(__name__)

# Hugging Face API Configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Get free token from huggingface.co
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

# Initialize or load model
model = None
vectorizer = None


def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def analyze_with_huggingface(text):
    """Use Hugging Face API for additional sentiment/credibility analysis"""
    if not HF_API_TOKEN or HF_API_TOKEN == "YOUR_HUGGING_FACE_API_TOKEN":
        return None

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    try:
        response = requests.post(HF_API_URL, headers=headers, json={"inputs": text[:512]})
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"HF API Error: {e}")

    return None


def scrape_news_content(url):
    """Simple web scraping for news articles"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            # Basic text extraction (you can enhance this with BeautifulSoup)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text[:5000]  # Limit to first 5000 chars
    except Exception as e:
        print(f"Scraping error: {e}")

    return None


def train_basic_model():
    """Train a basic fake news detection model"""
    # Sample training data (in real scenario, use a proper dataset)
    fake_samples = [
        "shocking discovery scientists baffled amazing miracle cure",
        "you won't believe what happened next click here now",
        "celebrities hate him for this one weird trick",
        "breaking government hiding the truth from you",
        "doctors stunned by this simple method"
    ]

    real_samples = [
        "according to the research published in nature journal",
        "the minister announced the new policy during press conference",
        "data from the national statistics office shows",
        "experts from the university conducted a study",
        "the company reported quarterly earnings today"
    ]

    texts = fake_samples + real_samples
    labels = [1] * len(fake_samples) + [0] * len(real_samples)

    global vectorizer, model
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X, labels)

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)


def load_model():
    """Load trained model"""
    global vectorizer, model

    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        train_basic_model()


def predict_fake_news(text):
    """Predict if news is fake or real"""
    cleaned_text = clean_text(text)

    # Transform text
    X = vectorizer.transform([cleaned_text])

    # Get prediction probability
    prediction = model.predict(X)[0]

    # Get probability scores
    try:
        proba = model.decision_function(X)[0]
        # Convert to percentage (0-100)
        # Normalize the score
        fake_percentage = 1 / (1 + np.exp(-proba))  # Sigmoid
        fake_percentage = fake_percentage * 100
    except:
        fake_percentage = prediction * 100

    return fake_percentage


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        url = data.get('url', '')

        if url:
            # Scrape content from URL
            scraped_text = scrape_news_content(url)
            if scraped_text:
                text = scraped_text
            else:
                return jsonify({'error': 'Could not scrape URL'}), 400

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Predict
        fake_percentage = predict_fake_news(text)

        # Optional: Use Hugging Face for additional analysis
        hf_result = analyze_with_huggingface(text)

        # Calculate final score
        real_percentage = 100 - fake_percentage

        result = {
            'fake_percentage': round(fake_percentage, 2),
            'real_percentage': round(real_percentage, 2),
            'verdict': 'LIKELY FAKE' if fake_percentage > 60 else 'LIKELY REAL' if fake_percentage < 40 else 'UNCERTAIN',
            'cleaned_text': clean_text(text)[:500] + '...' if len(text) > 500 else clean_text(text)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)