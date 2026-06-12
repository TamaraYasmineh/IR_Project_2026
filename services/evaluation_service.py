def precision_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):

    retrieved_docs = retrieved_docs[:k]

    relevant_found = 0

    for doc_id in retrieved_docs:

        if doc_id in relevant_docs:

            relevant_found += 1

    return relevant_found / k


def recall_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):

    retrieved_docs = retrieved_docs[:k]

    relevant_found = 0

    for doc_id in retrieved_docs:

        if doc_id in relevant_docs:

            relevant_found += 1

    return (
        relevant_found /
        len(relevant_docs)
    )


def average_precision(
    retrieved_docs,
    relevant_docs
):

    hits = 0

    sum_precision = 0

    for i, doc_id in enumerate(
        retrieved_docs,
        start=1
    ):

        if doc_id in relevant_docs:

            hits += 1

            sum_precision += (
                hits / i
            )

    if hits == 0:
        return 0

    return sum_precision / hits
import math


def ndcg_at_k(
    retrieved_docs,
    relevant_docs,
    k=10
):

    retrieved_docs = retrieved_docs[:k]

    dcg = 0

    for i, doc_id in enumerate(
        retrieved_docs,
        start=1
    ):

        relevance = (
            1
            if doc_id in relevant_docs
            else 0
        )

        dcg += (
            relevance /
            math.log2(i + 1)
        )

    ideal_relevances = [
        1
    ] * min(
        len(relevant_docs),
        k
    )

    idcg = 0

    for i, rel in enumerate(
        ideal_relevances,
        start=1
    ):

        idcg += (
            rel /
            math.log2(i + 1)
        )

    if idcg == 0:
        return 0

    return dcg / idcg