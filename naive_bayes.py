#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os

try:
    from IPython.display import display
except ImportError:
    display = print

dataset_path = "dataset/AI_Human.csv"
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "AI_Human.csv")

df = pd.read_csv(dataset_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
display(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# In[3]:


df = df.dropna()
df = df.drop_duplicates()

print("Shape after cleaning:", df.shape)
print("Missing values:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())


# In[4]:


print(df["generated"].value_counts())
print("\nPercentage distribution:")
print(df["generated"].value_counts(normalize=True) * 100)


# In[5]:


#Check the actual text data
for i in range(3):
    print(f"--- Sample {i+1} ---")
    print("Text:", df["text"].iloc[i])
    print("Label:", df["generated"].iloc[i])
    print()


# In[6]:


#Create a cleaned text column
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

print(df[["text", "clean_text"]].head(3))


# In[7]:


#removing the stop words 
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["clean_text"].apply(remove_stopwords)

print(df["clean_text"].head(3))


# In[8]:


#Separate Features and Target
X = df["clean_text"]
y = df["generated"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst text:")
print(X.iloc[0])

print("\nFirst label:")
print(y.iloc[0])


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nTraining label distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting label distribution:")
print(y_test.value_counts(normalize=True) * 100)


# In[10]:


#Create TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# In[11]:


print("Training matrix memory:",
      (X_train_tfidf.data.nbytes +
       X_train_tfidf.indices.nbytes +
       X_train_tfidf.indptr.nbytes) / (1024 ** 3),
      "GB")


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)

print(
    "Training matrix memory:",
    (
        X_train_tfidf.data.nbytes +
        X_train_tfidf.indices.nbytes +
        X_train_tfidf.indptr.nbytes
    ) / (1024 ** 3),
    "GB"
)


# In[20]:


from sklearn.naive_bayes import ComplementNB

cnb_model = ComplementNB()

cnb_model.fit(X_train_tfidf, y_train)

print("ComplementNB model trained successfully.")


# In[21]:


y_pred_cnb = cnb_model.predict(X_test_tfidf)

print("Predictions completed.")
print("Number of predictions:", len(y_pred_cnb))
print("First 10 predictions:", y_pred_cnb[:10])


# In[22]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred_cnb)
precision = precision_score(y_test, y_pred_cnb)
recall = recall_score(y_test, y_pred_cnb)
f1 = f1_score(y_test, y_pred_cnb)

print("ComplementNB Results")
print("--------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)


# In[23]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_cnb)

print("Confusion Matrix:")
print(cm)


# In[24]:


from sklearn.metrics import roc_auc_score

y_prob_cnb = cnb_model.predict_proba(X_test_tfidf)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob_cnb)

print("ComplementNB ROC-AUC:", roc_auc)


# In[26]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    "alpha": [0.1, 0.5, 1.0, 2.0]
}

grid_cnb = GridSearchCV(
    estimator=ComplementNB(),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=1
)

grid_cnb.fit(X_train_tfidf, y_train)

print("Best Alpha:", grid_cnb.best_params_["alpha"])
print("Best Cross-Validation F1:", grid_cnb.best_score_)


# In[27]:


best_cnb = grid_cnb.best_estimator_

y_pred_cnb_optimized = best_cnb.predict(X_test_tfidf)

print("Optimized ComplementNB predictions completed.")


# In[28]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_opt = accuracy_score(y_test, y_pred_cnb_optimized)
precision_opt = precision_score(y_test, y_pred_cnb_optimized)
recall_opt = recall_score(y_test, y_pred_cnb_optimized)
f1_opt = f1_score(y_test, y_pred_cnb_optimized)

print("Optimized ComplementNB Results")
print("-----------------------------")
print("Accuracy :", accuracy_opt)
print("Precision:", precision_opt)
print("Recall   :", recall_opt)
print("F1-Score :", f1_opt)


# In[29]:


from sklearn.metrics import roc_auc_score

y_prob_cnb_optimized = best_cnb.predict_proba(X_test_tfidf)[:, 1]

roc_auc_opt = roc_auc_score(
    y_test,
    y_prob_cnb_optimized
)

print("Optimized ComplementNB ROC-AUC:", roc_auc_opt)


# In[30]:


from sklearn.naive_bayes import MultinomialNB

mnb_model = MultinomialNB()

mnb_model.fit(X_train_tfidf, y_train)

print("MultinomialNB model trained successfully.")


# In[31]:


y_pred_mnb = mnb_model.predict(X_test_tfidf)

print("Predictions completed.")
print("Number of predictions:", len(y_pred_mnb))
print("First 10 predictions:", y_pred_mnb[:10])


# In[32]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
precision_mnb = precision_score(y_test, y_pred_mnb)
recall_mnb = recall_score(y_test, y_pred_mnb)
f1_mnb = f1_score(y_test, y_pred_mnb)

print("MultinomialNB Results")
print("---------------------")
print("Accuracy :", accuracy_mnb)
print("Precision:", precision_mnb)
print("Recall   :", recall_mnb)
print("F1-Score :", f1_mnb)


# In[33]:


from sklearn.metrics import roc_auc_score

y_prob_mnb = mnb_model.predict_proba(X_test_tfidf)[:, 1]

roc_auc_mnb = roc_auc_score(
    y_test,
    y_prob_mnb
)

print("MultinomialNB ROC-AUC:", roc_auc_mnb)


# In[34]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

param_grid_mnb = {
    "alpha": [0.1, 0.5, 1.0, 2.0]
}

grid_mnb = GridSearchCV(
    estimator=MultinomialNB(),
    param_grid=param_grid_mnb,
    cv=5,
    scoring="f1",
    n_jobs=1
)

grid_mnb.fit(X_train_tfidf, y_train)

print("Best Alpha:", grid_mnb.best_params_["alpha"])
print("Best Cross-Validation F1:", grid_mnb.best_score_)


# In[35]:


best_mnb = grid_mnb.best_estimator_

y_pred_mnb_optimized = best_mnb.predict(X_test_tfidf)

print("Optimized MultinomialNB predictions completed.")


# In[36]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_mnb_opt = accuracy_score(y_test, y_pred_mnb_optimized)
precision_mnb_opt = precision_score(y_test, y_pred_mnb_optimized)
recall_mnb_opt = recall_score(y_test, y_pred_mnb_optimized)
f1_mnb_opt = f1_score(y_test, y_pred_mnb_optimized)

print("Optimized MultinomialNB Results")
print("--------------------------------")
print("Accuracy :", accuracy_mnb_opt)
print("Precision:", precision_mnb_opt)
print("Recall   :", recall_mnb_opt)
print("F1-Score :", f1_mnb_opt)


# In[37]:


from sklearn.metrics import roc_auc_score

y_prob_mnb_optimized = best_mnb.predict_proba(X_test_tfidf)[:, 1]

roc_auc_mnb_opt = roc_auc_score(
    y_test,
    y_prob_mnb_optimized
)

print("Optimized MultinomialNB ROC-AUC:", roc_auc_mnb_opt)


# In[38]:


from sklearn.metrics import confusion_matrix

cm_cnb_opt = confusion_matrix(
    y_test,
    y_pred_cnb_optimized
)

print("Optimized ComplementNB Confusion Matrix:")
print(cm_cnb_opt)


# In[39]:


from sklearn.metrics import confusion_matrix

cm_mnb_opt = confusion_matrix(
    y_test,
    y_pred_mnb_optimized
)

print("Optimized MultinomialNB Confusion Matrix:")
print(cm_mnb_opt)


# In[40]:


try:
    from sklearn.metrics import roc_curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fpr_cnb, tpr_cnb, _ = roc_curve(y_test, y_prob_cnb_optimized)
    fpr_mnb, tpr_mnb, _ = roc_curve(y_test, y_prob_mnb_optimized)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr_cnb,
        tpr_cnb,
        label=f"ComplementNB (AUC = {roc_auc_opt:.4f})"
    )

    plt.plot(
        fpr_mnb,
        tpr_mnb,
        label=f"MultinomialNB (AUC = {roc_auc_mnb_opt:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: ComplementNB vs MultinomialNB")
    plt.legend()
    plt.grid()
    plt.savefig("roc_curve.png", bbox_inches="tight")
    plt.close()


    # In[42]:


    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, validation_scores = learning_curve(
        best_mnb,
        X_train_tfidf,
        y_train,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
        cv=3,
        scoring="f1",
        n_jobs=1
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training F1"
    )

    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation F1"
    )

    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve - Optimized MultinomialNB")
    plt.legend()
    plt.grid()
    plt.savefig("learning_curve.png", bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Plotting skipped: {e}")


# In[43]:


results = pd.DataFrame({
    "Model": [
        "ComplementNB",
        "MultinomialNB"
    ],
    
    "Best Alpha": [
        grid_cnb.best_params_["alpha"],
        grid_mnb.best_params_["alpha"]
    ],
    
    "CV F1": [
        grid_cnb.best_score_,
        grid_mnb.best_score_
    ],
    
    "Accuracy": [
        accuracy_opt,
        accuracy_mnb_opt
    ],
    
    "Precision": [
        precision_opt,
        precision_mnb_opt
    ],
    
    "Recall": [
        recall_opt,
        recall_mnb_opt
    ],
    
    "F1-Score": [
        f1_opt,
        f1_mnb_opt
    ],
    
    "ROC-AUC": [
        roc_auc_opt,
        roc_auc_mnb_opt
    ]
})

display(results.round(4))

