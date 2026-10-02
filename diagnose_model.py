import joblib
import pandas as pd
import numpy as np

tfidf = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("ai_human_detector_mnb.joblib")

test_inputs = [
    "This is a sentence I am writing completely by myself to test if the model detects human text.",
    "Furthermore, generative artificial intelligence leverages deep neural networks to produce coherent essays across diverse domains.",
    "I went to the market today to buy some fresh vegetables and fruits for dinner.",
    "Artificial intelligence is rapidly transforming modern technology and industries worldwide."
]

print("=== DIAGNOSING NAIVE BAYES PREDICTIONS ===")
for text in test_inputs:
    vec = tfidf.transform([text])
    active_features = vec.nonzero()[1]
    feature_names = tfidf.get_feature_names_out()
    matched_words = [feature_names[i] for i in active_features]
    
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    print(f"\nText: '{text}'")
    print(f"Matched TF-IDF Features ({len(matched_words)}): {matched_words}")
    print(f"Predicted Class: {'AI (1.0)' if pred == 1.0 else 'Human (0.0)'}")
    print(f"Probabilities -> Human (0.0): {prob[0]*100:.2f}%, AI (1.0): {prob[1]*100:.2f}%")
