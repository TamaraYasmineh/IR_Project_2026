import joblib
import pandas as pd

from services.tfidf_service import search_tfidf

print("Loading TF-IDF model...")

vectorizer = joblib.load(
    "models/tfidf/tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "models/tfidf/tfidf_matrix.pkl"
)

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

print("Model loaded successfully!")

query = input(
    "\nEnter your query: "
)

results, scores = search_tfidf(
    query,
    vectorizer,
    tfidf_matrix,
    top_k=10
)

print("\nTOP 10 RESULTS\n")

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