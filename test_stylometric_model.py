import os
import re
import time
import pandas as pd
import numpy as np
import joblib

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

print("=== STAGE 1: Loading Dataset ===")
dataset_path = "dataset/AI_Human.csv"
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "AI_Human.csv")

df = pd.read_csv(dataset_path).dropna().drop_duplicates()
print(f"Total dataset samples: {len(df):,}")

# Subsample 100,000 samples for fast experimental validation
df_sub = df.sample(n=100000, random_state=42)

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_sub["clean_text"] = df_sub["text"].apply(preprocess_text)

X = df_sub["clean_text"]
y = df_sub["generated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\n=== STAGE 2: Combining Word TF-IDF + Character N-Gram Features ===")
# Word TF-IDF (captures vocabulary & phrases)
word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=35000,
    sublinear_tf=True
)

# Char N-Gram TF-IDF (captures writing style, prefixes, suffixes, syntax independent of topic)
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=25000,
    sublinear_tf=True
)

union = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])

X_train_vec = union.fit_transform(X_train)
X_test_vec = union.transform(X_test)
print(f"Combined Feature Matrix Shape: {X_train_vec.shape}")

print("\n=== STAGE 3: Comparing Classifier Performance ===")

# Model A: MultinomialNB
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train_vec, y_train)
acc_mnb = accuracy_score(y_test, mnb.predict(X_test_vec))
print(f"MultinomialNB Accuracy: {acc_mnb*100:.2f}%")

# Model B: Logistic Regression (Stylometric & Calibrated Probabilities)
lr = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
lr.fit(X_train_vec, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_vec))
print(f"Logistic Regression Accuracy: {acc_lr*100:.2f}%")

print("\n=== STAGE 4: Real-World Generalization Test ===")
test_custom_sentences = [
    ("Human Casual", "I went to the market today to buy some fresh vegetables and fruits for dinner."),
    ("Human Personal", "Honestly I think studying late at night works better for me because there are fewer distractions."),
    ("AI ChatGPT", "Furthermore, it is essential to consider the multifaceted implications of climate change on urban infrastructure."),
    ("AI Explanatory", "In conclusion, artificial intelligence algorithms offer unprecedented opportunities for data optimization.")
]

for desc, text in test_custom_sentences:
    cleaned = preprocess_text(text)
    vec = union.transform([cleaned])
    
    # MNB Prediction
    mnb_pred = mnb.predict(vec)[0]
    mnb_prob = mnb.predict_proba(vec)[0]
    
    # LR Prediction
    lr_pred = lr.predict(vec)[0]
    lr_prob = lr.predict_proba(vec)[0]
    
    print(f"\n[{desc}] Text: '{text}'")
    print(f"  Naive Bayes:   {'AI' if mnb_pred==1 else 'Human'} (Human: {mnb_prob[0]*100:.1f}%, AI: {mnb_prob[1]*100:.1f}%)")
    print(f"  Logistic Reg:  {'AI' if lr_pred==1 else 'Human'} (Human: {lr_prob[0]*100:.1f}%, AI: {lr_prob[1]*100:.1f}%)")
