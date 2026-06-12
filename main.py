import ir_datasets

dataset = ir_datasets.load("lotte/lifestyle/dev/search")

print("\n=== FIRST DOCUMENT ===\n")

doc = next(dataset.docs_iter())

print("DOC ID:")
print(doc.doc_id)

print("\nTEXT:")
print(doc.text[:1000])