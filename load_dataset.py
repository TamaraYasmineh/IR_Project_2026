import ir_datasets

print("Loading dataset...")

dataset = ir_datasets.load("beir/quora")

print("Dataset loaded successfully!")

print("\nNumber of Queries:")
print(dataset.queries_count())

print("\nNumber of Documents:")
print(dataset.docs_count())

print("\nFirst Query:")
query = next(dataset.queries_iter())
print(query)

print("\nFirst Document:")
doc = next(dataset.docs_iter())
print(doc)