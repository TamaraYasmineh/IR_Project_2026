from rank_bm25 import BM25Okapi


def build_bm25_model(
    tokenized_documents,
    k1=1.5,
    b=0.75
):
    bm25 = BM25Okapi(
        tokenized_documents,
        k1=k1,
        b=b
    )

    return bm25


def search_bm25(
    query_tokens,
    bm25_model,
    top_k=10
):

    scores = bm25_model.get_scores(
        query_tokens
    )

    ranked_indices = scores.argsort()[::-1]

    filtered_indices = [
        idx
        for idx in ranked_indices
        if scores[idx] > 0
    ]

    return filtered_indices[:top_k], scores