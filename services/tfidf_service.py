from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_model(documents):

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    return vectorizer, tfidf_matrix


def search_tfidf(
    query,
    vectorizer,
    tfidf_matrix,
    top_k=10
):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    ranked_indices = similarities.argsort()[::-1]

    return ranked_indices[:top_k], similarities