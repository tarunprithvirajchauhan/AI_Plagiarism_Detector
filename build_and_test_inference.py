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
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

print("=== STAGE 1: Loading & Cleaning Dataset ===")
dataset_path = "dataset/AI_Human.csv"
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "AI_Human.csv")

df = pd.read_csv(dataset_path)
df = df.dropna().drop_duplicates()

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

print(f"Total samples after cleaning: {len(df):,}")
print("Cleaning text column...")
df["clean_text"] = df["text"].apply(preprocess_text)

X = df["clean_text"]
y = df["generated"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train):,}")
print(f"Testing set size:  {len(X_test):,}")

print("\n=== STAGE 2: TF-IDF Vectorization ===")
start_v = time.time()
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF feature extraction completed in {time.time()-start_v:.2f} seconds.")

print("\n=== STAGE 3: Model Training & Evaluation ===")
# Model 1: Optimized MultinomialNB (alpha=0.1)
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train_tfidf, y_train)

y_pred_mnb = mnb.predict(X_test_tfidf)
y_prob_mnb = mnb.predict_proba(X_test_tfidf)[:, 1]

acc_mnb = accuracy_score(y_test, y_pred_mnb)
auc_mnb = roc_auc_score(y_test, y_prob_mnb)

print("\n-------------------------------------------")
print(" MULTINOMIAL NAIVE BAYES (alpha = 0.1)")
print("-------------------------------------------")
print(f"Overall Accuracy: {acc_mnb * 100:.2f}%")
print(f"ROC-AUC Score:    {auc_mnb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_mnb, target_names=["Human (0)", "AI (1)"], digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mnb))

# Model 2: Optimized ComplementNB (alpha=0.1)
cnb = ComplementNB(alpha=0.1)
cnb.fit(X_train_tfidf, y_train)

y_pred_cnb = cnb.predict(X_test_tfidf)
y_prob_cnb = cnb.predict_proba(X_test_tfidf)[:, 1]

acc_cnb = accuracy_score(y_test, y_pred_cnb)
auc_cnb = roc_auc_score(y_test, y_prob_cnb)

print("\n-------------------------------------------")
print(" COMPLEMENT NAIVE BAYES (alpha = 0.1)")
print("-------------------------------------------")
print(f"Overall Accuracy: {acc_cnb * 100:.2f}%")
print(f"ROC-AUC Score:    {auc_cnb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cnb, target_names=["Human (0)", "AI (1)"], digits=4))

print("\n=== STAGE 4: Saving Model for Web Application Use ===")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
joblib.dump(mnb, "ai_human_detector_mnb.joblib")
print("Saved 'tfidf_vectorizer.joblib' and 'ai_human_detector_mnb.joblib' successfully!")

print("\n=== STAGE 5: Live Real-World Inference Test ===")
sample_texts = [
    {
        "desc": "Real Human Essay sample (Cars in cities)",
        "text": "Cars have been around since Henry Ford created the Model T. They played a major role in our daily lives. But limiting car use might be a good thing to reduce air pollution and traffic jams in big cities."
    },
    {
        "desc": "AI Generated Essay sample (ChatGPT style)",
        "text": "In recent years, artificial intelligence has reshaped modern technology. From healthcare automation to generative language models, machine learning systems provide unparalleled efficiency and data synthesis capabilities."
    },
    {
        "desc": "Short Human sentence",
        "text": "I went to the store yesterday and bought some apples and milk for breakfast."
    },
    {
        "desc": "AI Explanatory text",
        "text": "Furthermore, it is essential to consider the multifaceted implications of climate policy. Consequently, sustainable urbanization strategies must incorporate public transportation infrastructure."
    }
]

def predict_single(raw_text):
    t0 = time.time()
    cleaned = preprocess_text(raw_text)
    vec = tfidf.transform([cleaned])
    pred_label = mnb.predict(vec)[0]
    prob = mnb.predict_proba(vec)[0]
    dt = (time.time() - t0) * 1000
    label_str = "AI-Generated (1.0)" if pred_label == 1.0 else "Human-Written (0.0)"
    confidence = prob[int(pred_label)] * 100
    return label_str, confidence, dt

print("\n--- Testing Live Predictions ---")
for idx, s in enumerate(sample_texts, 1):
    label_str, conf, dt = predict_single(s["text"])
    print(f"Sample {idx} [{s['desc']}]:")
    print(f"  Prediction : {label_str}")
    print(f"  Confidence : {conf:.2f}%")
    print(f"  Latency    : {dt:.2f} ms\n")
