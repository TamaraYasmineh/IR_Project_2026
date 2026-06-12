import pandas as pd

from services.bm25_service import (
    build_bm25_model,
    search_bm25
)

print("Loading dataset...")

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

documents = df["processed_text"].tolist()

print("Tokenizing documents...")

tokenized_documents = [
    doc.split()
    for doc in documents
]

print("Building BM25 model...")

bm25_model = build_bm25_model(
    tokenized_documents
)

print("BM25 Ready!")

query = input(
    "\nEnter Query: "
)

query_tokens = query.split()

results, scores = search_bm25(
    query_tokens,
    bm25_model,
    top_k=10
)

print("\nTOP RESULTS\n")

for rank, idx in enumerate(results, start=1):

    print(f"Rank #{rank}")

    print(
        "Score:",
        round(float(scores[idx]), 4)
    )

    print(
        df.iloc[idx]["original_text"]
    )

    print("-" * 60)