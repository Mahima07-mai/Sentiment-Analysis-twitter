import streamlit as st
import joblib
import re
import emoji
import string
import nltk
import os
from nltk.corpus import stopwords

# ---------- Load artifacts ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "models", "model_v2.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "models", "vectorizer_v2.pkl"))

# ---------- Preprocessing ----------
stop_words = set(stopwords.words('english'))
negations = {"no", "not", "nor", "never", "n't"}
stop_words = stop_words - negations

def clean_text(text):
    text = str(text).lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z!? ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

    
# ---------- UI ----------
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet and predict sentiment")

user_input = st.text_area("Tweet text")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Enter some text")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        st.subheader("Predicted Sentiment:")
        st.success(prediction)
