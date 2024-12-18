import openai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# Configuration
openai.api_key = 'your-openai-api-key'

# Initialize Flask app
app = Flask(__name__)

# Dummy user data processing
def process_user_data():
    # Load or fetch user interaction data
    data = pd.read_csv("user_interactions.csv")
    return data

# Content Recommendation
def recommend_content(user_id, data):
    # Extract user data
    user_data = data[data['user_id'] == user_id]
    # Use TF-IDF vectorization for recommendation
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['content'])
    similarities = cosine_similarity(tfidf_matrix)
    return similarities[user_id].argsort()[-10:]  # Recommend top 10

# Content Summarization using GPT
def summarize_content(content):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Summarize the following text: {content}",
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()

# Sentiment Analysis
def analyze_sentiment(content):
    # Placeholder for sentiment analysis logic
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(content)

# API Endpoint (Flask)
@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    data = process_user_data()
    recommendations = recommend_content(user_id, data)
    return jsonify(recommendations=list(recommendations))

@app.route('/summarize', methods=['POST'])
def summarize():
    content = request.json['content']
    summary = summarize_content(content)
    return jsonify(summary=summary)

@app.route('/sentiment', methods=['POST'])
def sentiment():
    content = request.json['content']
    sentiment = analyze_sentiment(content)
    return jsonify(sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
