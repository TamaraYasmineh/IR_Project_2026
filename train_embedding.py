import pandas as pd
import joblib

from services.embedding_service import (
    build_embedding_model
)

print("Loading processed dataset...")

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

documents = df["processed_text"].tolist()

print("Building Embedding model...")

model, embeddings = build_embedding_model(
    documents
)

print("Saving embeddings...")

joblib.dump(
    embeddings,
    "models/embeddings/embeddings.pkl"
)

print("Embeddings saved successfully!")