import pandas as pd
import nltk
import re
import emoji
import os
from nltk.corpus import stopwords

# Download once
nltk.download('stopwords')
nltk.download('punkt')

# Stopwords (keep negations)
stop_words = set(stopwords.words('english'))
negations = {"no", "not", "nor", "never", "n't"}
stop_words = stop_words - negations

def clean_tweet(text):
    text = str(text).lower()

    # Convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Keep hashtag text
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove unwanted characters (keep ! ?)
    text = re.sub(r"[^a-z!? ]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    return " ".join(tokens)
cols = ['id', 'topic', 'sentiment', 'text']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "twitter_training.csv"),names=cols,header=None) 


# Keep only needed columns
df = df[['sentiment', 'text']].dropna()

# Drop Irrelevant class
df = df[df['sentiment'] != 'Irrelevant']

# Clean text
df['clean_text'] = df['text'].apply(clean_tweet)

# Remove empty rows
df = df[df['clean_text'].str.len() > 0]

# Save
df.to_csv('D://Sentiment - Project/data/clean_twitter_v2.csv',index=False)

print("Preprocessing complete")
