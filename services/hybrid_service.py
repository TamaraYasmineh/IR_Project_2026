import numpy as np


def hybrid_search(
    tfidf_scores,
    bm25_scores,
    alpha=0.5,
    top_k=10
):

    tfidf_scores = np.array(tfidf_scores)
    bm25_scores = np.array(bm25_scores)

    if tfidf_scores.max() > 0:
        tfidf_scores = (
            tfidf_scores /
            tfidf_scores.max()
        )

    if bm25_scores.max() > 0:
        bm25_scores = (
            bm25_scores /
            bm25_scores.max()
        )

    hybrid_scores = (
        alpha * tfidf_scores +
        (1 - alpha) * bm25_scores
    )

    ranked_indices = (
        hybrid_scores.argsort()[::-1]
    )

    return (
        ranked_indices[:top_k],
        hybrid_scores
    )