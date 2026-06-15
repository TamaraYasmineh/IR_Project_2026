import ir_datasets

dataset = ir_datasets.load(
    "beir/quora/dev"
)

print("Dataset Loaded")

print("\nQueries:")
print(dataset.queries_count())

print("\nFirst Query:")
print(next(dataset.queries_iter()))

print("\nFirst Qrel:")
print(next(dataset.qrels_iter()))