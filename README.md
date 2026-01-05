# Twitter Sentiment Analysis using Machine Learning

An end-to-end **Natural Language Processing (NLP)** project that classifies tweets into **Positive, Negative, or Neutral** sentiments using **TF-IDF features** and **Logistic Regression**, with a **Streamlit-based web application** for real-time sentiment inference.

---

## 📌 Project Overview

Twitter generates massive amounts of unstructured text data every day. Extracting sentiment from this data is useful for:

- Brand and product monitoring  
- Opinion mining  
- Public sentiment analysis  
- Market research  

This project implements a **complete machine learning pipeline** starting from raw tweet data and ending with a deployable web application.

---

## 🧠 Problem Statement

Given a tweet, predict its sentiment category:

- **Positive**  
- **Negative**  
- **Neutral**  

This is a **multiclass text classification problem** on noisy social media data.

---

## 🧪 Dataset

- **Source:** Twitter sentiment dataset  
- **Columns used:**  
  - `text` – raw tweet  
  - `sentiment` – target label  
- The `Irrelevant` class is removed to focus on meaningful sentiment categories.

---

## ⚙️ Text Preprocessing

Tweets contain noise such as emojis, hashtags, mentions, URLs, and informal language.  
The preprocessing pipeline includes:

- Convert text to lowercase  
- Convert emojis to textual meaning (e.g., 😊 → `smiling_face`)  
- Remove URLs and user mentions  
- Preserve hashtag words (remove `#`, keep text)  
- Remove unwanted characters (keep `!` and `?` for sentiment)  
- Normalize whitespace  
- Remove stopwords while preserving negations (`not`, `no`, `never`)  

This ensures sentiment-relevant information is retained.

---

## 📊 Feature Engineering (TF-IDF)

Text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

**Configuration:**

- Unigrams and bigrams (`ngram_range=(1,2)`)  
- Maximum features limited to 20,000  
- Rare and overly common words filtered  
- Sublinear term frequency scaling applied  

TF-IDF is effective for classical ML models on text data.

---

## 🏋️ Model Training

- **Model:** Logistic Regression  
- **Solver:** `lbfgs` (supports multiclass softmax)  
- **Class imbalance handling:** `class_weight='balanced'`  
- **Train/Test split:** 80/20 with stratification  

**Why Logistic Regression?**

- Strong baseline performance  
- Fast training and inference  
- Interpretability  
- Proven effectiveness with TF-IDF features

---

## 📈 Model Performance

- **Accuracy:** ~93%
- **Class-wise F1-scores:** 
- **Negative:** 0.91
- **Neutral:** 0.93
- **Positive:** 0.92


The model achieves balanced performance across all sentiment classes.

---

## 🖥️ Streamlit Web Application

A **Streamlit UI** is implemented for real-time sentiment prediction.

**Features:**

- User inputs a tweet  
- Text is preprocessed using the same pipeline as training  
- Model predicts sentiment instantly  
- Clean and simple interface  

This demonstrates how the model can be used in real-world applications.

---


## 🚀 How to Run the Project Locally

### 1️⃣ Clone the repository
```
git clone https://github.com/your-username/sentiment-analysis-twitter.git
cd sentiment-analysis-twitter
```
(Replace your-username with the actual repository link)

### 2️⃣ Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # macOS/Linux
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Run preprocessing
```
python src/preprocessing.py
```

### 5️⃣ Train the model
```
python src/model.py
```

### 6️⃣ Run the Streamlit app
```
streamlit run app.py
```

## 🔮 Future Improvements

- Use transformer-based models (BERT, RoBERTa)
- Hyperparameter tuning with GridSearchCV
- Add confidence scores for predictions
- Deploy on Streamlit Cloud or AWS
- Extend to multilingual sentiment analysis

## 🧾 License 
This project is licensed under MIT License - see the LICENSE file for details

## 👤 Author
- Mahima A
- Second-year undergraduate student
- Interested in Machine Learning, NLP, and AI applications