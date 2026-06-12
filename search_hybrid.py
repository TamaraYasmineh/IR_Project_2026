import joblib
import pandas as pd

from services.tfidf_service import search_tfidf

from services.bm25_service import (
    build_bm25_model,
    search_bm25
)

from services.hybrid_service import (
    hybrid_search
)

print("Loading data...")

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

documents = df["processed_text"].tolist()

tokenized_documents = [
    doc.split()
    for doc in documents
]

vectorizer = joblib.load(
    "models/tfidf/tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "models/tfidf/tfidf_matrix.pkl"
)

bm25_model = build_bm25_model(
    tokenized_documents
)

query = input(
    "\nEnter query: "
)

_, tfidf_scores = search_tfidf(
    query,
    vectorizer,
    tfidf_matrix,
    top_k=len(documents)
)

query_tokens = query.split()

_, bm25_scores = search_bm25(
    query_tokens,
    bm25_model,
    top_k=len(documents)
)

results, hybrid_scores = hybrid_search(
    tfidf_scores,
    bm25_scores,
    alpha=0.5,
    top_k=10
)

print("\nTOP HYBRID RESULTS\n")

for rank, idx in enumerate(results, start=1):

    print(f"Rank #{rank}")

    print(
        "Score:",
        round(float(hybrid_scores[idx]), 4)
    )

    print(
        df.iloc[idx]["original_text"]
    )

    print("-" * 60)