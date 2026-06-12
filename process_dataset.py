import ir_datasets
import pandas as pd

from services.preprocessing_service import preprocess_text

print("Loading Quora dataset...")

dataset = ir_datasets.load("beir/quora")

processed_docs = []

print("Processing first 1000 documents...")

for i, doc in enumerate(dataset.docs_iter()):

    cleaned_text = preprocess_text(doc.text)

    processed_docs.append({
        "doc_id": doc.doc_id,
        "original_text": doc.text,
        "processed_text": " ".join(cleaned_text)
    })

    if i >= 999:
        break

print("Saving CSV...")

df = pd.DataFrame(processed_docs)

df.to_csv(
    "data/processed/quora_processed_1000.csv",
    index=False
)

print("Done!")
print("Documents processed:", len(df))