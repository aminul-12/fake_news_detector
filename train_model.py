"""
train_model.py

Loads Fake.csv and True.csv from ./data, combines them, preprocesses text, vectorizes using TF-IDF,
trains a Logistic Regression classifier, evaluates it, and saves the model and vectorizer to model.joblib.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from utils import clean_text, ensure_nltk

def load_data(data_folder="data"):
    fake_path = os.path.join(data_folder, "Fake.csv")
    true_path = os.path.join(data_folder, "True.csv")
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError("Place 'Fake.csv' and 'True.csv' inside the 'data' folder.")
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake["label"] = 0
    true["label"] = 1
    # many versions of these datasets use columns 'title' and 'text'
    # we'll combine title + text for input
    def merge_columns(row):
        parts = []
        for col in ["title","text"]:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        return " ".join(parts)
    fake["content"] = fake.apply(merge_columns, axis=1)
    true["content"] = true.apply(merge_columns, axis=1)
    df = pd.concat([fake, true], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df[["content","label"]]

def main():
    ensure_nltk()
    print("Loading data...")
    df = load_data()
    print("Cleaning text...")
    df["content"] = df["content"].fillna("").apply(clean_text)
    X = df["content"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1,2), stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    print("Evaluating...")
    preds = clf.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    # Save model and vectorizer
    joblib.dump({"model": clf, "vectorizer": tfidf}, "model.joblib")
    print("Saved model.joblib")

if __name__ == "__main__":
    main()