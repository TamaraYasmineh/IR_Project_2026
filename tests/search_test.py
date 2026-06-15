import json
import pandas as pd

# تحميل الـ Index
with open(
    "data/indexes/inverted_index.json",
    "r",
    encoding="utf-8"
) as f:
    index = json.load(f)

# تحميل الوثائق الأصلية
df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

query = input("Enter search term: ").lower()

if query in index:

    doc_ids = index[query]

    print("\nDocuments Found:", len(doc_ids))
    print("\nTop Results:\n")

    for doc_id in doc_ids[:10]:

        doc = df[df["doc_id"] == doc_id]

        if not doc.empty:
            print(f"Document ID: {doc_id}")
            print(doc.iloc[0]["original_text"])
            print("-" * 60)

else:

    print("No results found.")