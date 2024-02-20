import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Replace 'YOUR_API_KEY' with your actual NewsAPI key
API_KEY = '1ea4129682b54fc59f1f58846490d2a3'

# Sample user preferences (keywords of interest)
user_preferences = ['technology', 'politics', 'sports']

# Fetch news articles from multiple sources using NewsAPI
def fetch_news():
    sources = ['bbc-news', 'cnn', 'reuters', 'al-jazeera-english']
    all_articles = []

    for source in sources:
        url = f'https://newsapi.org/v2/top-headlines?sources={source}&apiKey={API_KEY}'
        response = requests.get(url)
        data = response.json()
        articles = data.get('articles', [])
        all_articles.extend(articles)

    return all_articles

# Sample user-article interaction matrix (user_id, article_id, interaction_strength)
interaction_data = {
    'user_id': [1, 1, 2, 2, 3],
    'article_id': [0, 1, 1, 2, 3],
    'interaction_strength': [2, 3, 1, 2, 3]
}

interaction_df = pd.DataFrame(interaction_data)

# Fetch news articles
news_articles = fetch_news()

# Extract text content from the articles
article_texts = [article['title'] + ' ' + article['description'] for article in news_articles]

# Calculate TF-IDF vectors for the articles
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(article_texts)

# Calculate cosine similarity between user preferences and article content
user_profile = ' '.join(user_preferences)
user_tfidf = tfidf_vectorizer.transform([user_profile])
cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix)

# Get personalized recommendations based on user interactions and preferences
user_id = 1
user_interactions = interaction_df[interaction_df['user_id'] == user_id]

# Calculate the user's aggregated interaction scores for each article
user_interactions_grouped = user_interactions.groupby('article_id')['interaction_strength'].sum().reset_index()

# Sort articles by the calculated interaction score
user_interactions_grouped = user_interactions_grouped.sort_values(by='interaction_strength', ascending=False)

# Get recommended articles based on cosine similarities and user interactions
recommended_articles = [news_articles[i] for i in user_interactions_grouped['article_id']]

# Display recommended articles
print("Personalized News Recommendations:")
for idx, article in enumerate(recommended_articles):
    print(f"{idx + 1}. {article['title']}")
    print(f"Source: {article['source']['name']}")
    print(f"Description: {article['description']}")
    print(f"URL: {article['url']}")
    print()
