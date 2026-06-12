from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def build_embedding_model(documents):

    model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2"
   )
    embeddings = model.encode(
        documents,
        show_progress_bar=True
    )

    return model, embeddings


def search_embeddings(
    query,
    model,
    embeddings,
    top_k=10
):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    ).flatten()

    ranked_indices = similarities.argsort()[::-1]

    return ranked_indices[:top_k], similarities