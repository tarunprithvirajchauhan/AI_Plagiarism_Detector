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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

print("=== STAGE 1: Loading Full Dataset ===")
dataset_path = "dataset/AI_Human.csv"
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "AI_Human.csv")

df = pd.read_csv(dataset_path).dropna().drop_duplicates()
print(f"Total dataset samples: {len(df):,}")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

print("Preprocessing text...")
df["clean_text"] = df["text"].apply(preprocess_text)

X = df["clean_text"]
y = df["generated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")

print("\n=== STAGE 2: Building Sublinear TF-IDF Vectorizer ===")
start_t = time.time()
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_features=60000,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF Vectorization completed in {time.time() - start_t:.2f} seconds.")

print("\n=== STAGE 3: Training Calibrated Classifier ===")
# Calibrated Logistic Regression provides robust, accurate probabilities across any user text
clf = LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)
y_prob = clf.predict_proba(X_test_tfidf)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n-------------------------------------------")
print(" CALIBRATED MODEL PERFORMANCE RESULTS")
print("-------------------------------------------")
print(f"Overall Testing Accuracy: {acc * 100:.2f}%")
print(f"ROC-AUC Score:            {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"], digits=4))

print("\n=== STAGE 4: Saving Calibrated Binaries ===")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
joblib.dump(clf, "ai_human_detector_mnb.joblib")
print("Updated 'tfidf_vectorizer.joblib' and 'ai_human_detector_mnb.joblib' with calibrated binaries!")

print("\n=== STAGE 5: Testing Real-World Custom Sentences ===")
test_sentences = [
    ("User Typed Human Sentence 1", "This is a sentence I typed by myself to test if the detector works properly."),
    ("User Typed Human Sentence 2", "I went to the store today to buy some fresh fruits and vegetables for dinner."),
    ("User Typed Human Casual", "Honestly I feel like studying late at night works better for me because there are fewer distractions."),
    ("AI ChatGPT Text 1", "Artificial intelligence has transformed modern technology through rapid advancements in natural language processing."),
    ("AI ChatGPT Text 2", "Furthermore, it is essential to consider the multifaceted implications of climate change on urban infrastructure.")
]

for desc, text in test_sentences:
    cleaned = preprocess_text(text)
    vec = tfidf.transform([cleaned])
    pred = clf.predict(vec)[0]
    prob = clf.predict_proba(vec)[0]
    
    is_ai = (pred == 1.0)
    conf = prob[int(pred)] * 100
    
    print(f"\n[{desc}]")
    print(f"  Text: '{text}'")
    print(f"  Prediction : {'AI-Generated' if is_ai else 'Human-Written'}")
    print(f"  Confidence : {conf:.2f}% (Human: {prob[0]*100:.2f}%, AI: {prob[1]*100:.2f}%)")
