import pandas as pd
import joblib

from services.tfidf_service import build_tfidf_model

print("Loading processed dataset...")

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

documents = df["processed_text"].tolist()

print("Training TF-IDF model...")

vectorizer, tfidf_matrix = build_tfidf_model(
    documents
)

print("Saving model...")

joblib.dump(
    vectorizer,
    "models/tfidf/tfidf_vectorizer.pkl"
)

joblib.dump(
    tfidf_matrix,
    "models/tfidf/tfidf_matrix.pkl"
)

print("TF-IDF model saved successfully!")

print("Vectorizer saved:")
print("models/tfidf/tfidf_vectorizer.pkl")

print("Matrix saved:")
print("models/tfidf/tfidf_matrix.pkl")