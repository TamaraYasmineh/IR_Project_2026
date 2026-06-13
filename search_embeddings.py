import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer

from services.embedding_service import (
    search_embeddings
)

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

print("Loading embeddings...")

embeddings = joblib.load(
    "models/embeddings/embeddings.pkl"
)

print("Loading model...")

model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2"
)

query = input(
    "\nEnter Query: "
)

results, scores = search_embeddings(
    query,
    model,
    embeddings,
    top_k=10
)

print("\nTOP EMBEDDING RESULTS\n")

for rank, idx in enumerate(
    results,
    start=1
):

    print(f"Rank #{rank}")

    print(
        "Score:",
        round(float(scores[idx]),4)
    )

    print(
        df.iloc[idx]["original_text"]
    )

    print("-" * 60)