import pandas as pd

from services.tfidf_service import (
    build_tfidf_model,
    search_tfidf
)

print("Loading processed dataset...")

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

documents = df["processed_text"].tolist()

print("Building TF-IDF model...")

vectorizer, tfidf_matrix = build_tfidf_model(
    documents
)

print("Model Ready!")

query = input(
    "\nEnter your query: "
)

results, scores = search_tfidf(
    query,
    vectorizer,
    tfidf_matrix,
    top_k=10
)

print("\nTOP RESULTS\n")

for rank, idx in enumerate(results, start=1):

    print(f"Rank #{rank}")

    print(
        "Score:",
        round(scores[idx], 4)
    )

    print(
        df.iloc[idx]["original_text"]
    )

    print("-" * 60)