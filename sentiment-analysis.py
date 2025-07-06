# sentiment_analysis.py

import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords

nltk.download('stopwords')

# Step 1: Load dataset
df = pd.read_csv("data/tweets.csv")
df = df[['text', 'airline_sentiment']]
df.columns = ['text', 'sentiment']
print(df.head())

# Step 2: Clean the tweets
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: Use TextBlob to get polarity and sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['predicted_sentiment'] = df['clean_text'].apply(get_sentiment)

# Step 4: Visualization
# Sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='predicted_sentiment', data=df, palette='cool')
plt.title("Predicted Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Count")
plt.show()

# WordClouds
def show_wordcloud(sentiment):
    text = " ".join(review for review in df[df['predicted_sentiment'] == sentiment]['clean_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{sentiment} Tweets WordCloud")
    plt.show()

show_wordcloud("Positive")
show_wordcloud("Negative")
show_wordcloud("Neutral")

# Step 5: Accuracy check (Optional)
from sklearn.metrics import classification_report

print("Accuracy Report (Predicted vs Original Sentiment)")
print(classification_report(df['sentiment'], df['predicted_sentiment']))
