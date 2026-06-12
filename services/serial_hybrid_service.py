import numpy as np


def serial_hybrid_search(
    tfidf_scores,
    bm25_scores,
    candidate_size=50,
    top_k=10
):

    candidate_docs = np.argsort(
        tfidf_scores
    )[::-1][:candidate_size]

    candidate_scores = []

    for doc_id in candidate_docs:

        candidate_scores.append(
            (
                doc_id,
                bm25_scores[doc_id]
            )
        )

    candidate_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    final_docs = [
        doc_id
        for doc_id, _
        in candidate_scores[:top_k]
    ]

    return final_docs