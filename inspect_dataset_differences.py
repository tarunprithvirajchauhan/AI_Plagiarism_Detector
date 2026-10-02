import pandas as pd
import numpy as np

df = pd.read_csv("dataset/AI_Human.csv").dropna().drop_duplicates()

df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
df["char_count"] = df["text"].apply(lambda x: len(str(x)))

human = df[df["generated"] == 0.0]
ai = df[df["generated"] == 1.0]

print("=== DATASET METRICS ANALYSIS ===")
print("Human text count:", len(human))
print("AI text count:   ", len(ai))

print("\n--- Word Count Statistics ---")
print("Human Avg Word Count:", human["word_count"].mean())
print("Human Min Word Count:", human["word_count"].min())
print("Human Max Word Count:", human["word_count"].max())

print("\nAI Avg Word Count:", ai["word_count"].mean())
print("AI Min Word Count:", ai["word_count"].min())
print("AI Max Word Count:", ai["word_count"].max())

print("\n--- First 3 Human Samples ---")
for i in range(3):
    print(f"Sample {i+1} ({human['word_count'].iloc[i]} words):")
    print(human["text"].iloc[i][:250] + "...\n")

print("--- First 3 AI Samples ---")
for i in range(3):
    print(f"Sample {i+1} ({ai['word_count'].iloc[i]} words):")
    print(ai["text"].iloc[i][:250] + "...\n")
