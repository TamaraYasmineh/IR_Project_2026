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

tokenized_documents = [
    doc.split()
    for doc in documents
]

query = input(
    "\nEnter Query: "
)

k1 = float(
    input("Enter k1 value: ")
)

b = float(
    input("Enter b value: ")
)

print("\nBuilding BM25 model...")

bm25_model = build_bm25_model(
    tokenized_documents,
    k1=k1,
    b=b
)

query_tokens = query.split()

results, scores = search_bm25(
    query_tokens,
    bm25_model,
    top_k=10
)
print("DEBUG RESULTS:", len(results))
print("\n===================================")
print("BM25 PARAMETERS")
print("===================================")

print(f"k1 = {k1}")
print(f"b = {b}")

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