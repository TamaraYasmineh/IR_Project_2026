import pandas as pd
import joblib

from services.bm25_service import build_bm25_model

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

documents = df["processed_text"].tolist()

tokenized_documents = [
    doc.split()
    for doc in documents
]

print("Training BM25 model...")

bm25_model = build_bm25_model(
    tokenized_documents,
    k1=1.5,
    b=0.75
)

print("Saving model...")

joblib.dump(
    bm25_model,
    "models/bm25/bm25_model.pkl"
)

print("BM25 model saved successfully!")