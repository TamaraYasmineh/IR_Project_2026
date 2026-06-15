from services.evaluation_service import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k
)

retrieved_docs = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10
]

relevant_docs = [
    2, 4, 6, 8, 10
]

precision = precision_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
)

recall = recall_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
)

print(
    "Precision@10 =",
    round(precision, 4)
)

print(
    "Recall@10 =",
    round(recall, 4)
)
ap = average_precision(
    retrieved_docs,
    relevant_docs
)

print(
    "Average Precision =",
    round(ap, 4)
)
ndcg = ndcg_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
)

print(
    "nDCG@10 =",
    round(ndcg, 4)
)