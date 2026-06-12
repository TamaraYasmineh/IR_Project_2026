import pandas as pd
import json
import os

from services.indexing_service import build_inverted_index

print("=" * 50)
print("LOADING PROCESSED DATASET")
print("=" * 50)

# تحميل البيانات المعالجة
df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)

print(f"Documents Loaded: {len(df)}")

print("\n" + "=" * 50)
print("BUILDING INVERTED INDEX")
print("=" * 50)

# بناء الـ Index
index = build_inverted_index(df)

print(f"Unique Terms: {len(index)}")

print("\n" + "=" * 50)
print("SAMPLE TERM TEST")
print("=" * 50)

term = "india"

if term in index:
    print(f"{term} -> {index[term][:20]}")
else:
    print(f"'{term}' not found")

print("\n" + "=" * 50)
print("SAVING INDEX")
print("=" * 50)

# إنشاء المجلد إذا لم يكن موجوداً
os.makedirs(
    "data/indexes",
    exist_ok=True
)

# حفظ الـ Index
with open(
    "data/indexes/inverted_index.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        index,
        f,
        ensure_ascii=False,
        indent=4
    )

print("Index saved successfully!")

print("\nSaved File:")
print("data/indexes/inverted_index.json")

print("\nDONE!")