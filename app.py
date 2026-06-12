import streamlit as st
import joblib
import pandas as pd
from services.tfidf_service import search_tfidf
from services.bm25_service import (
    build_bm25_model,
    search_bm25
)
from services.hybrid_service import (
    hybrid_search
)
from services.serial_hybrid_service import (
    serial_hybrid_search
)
from services.query_refinement_service import (
    refine_query
)
st.set_page_config(
    page_title="IR Search System",
    page_icon="🔍",
    layout="wide"
)
from services.evaluation_service import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k
)
st.title("🔍 Information Retrieval System")
st.sidebar.header("Search Parameters")

k1 = st.sidebar.slider(
    "BM25 k1",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1
)

b = st.sidebar.slider(
    "BM25 b",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

alpha = st.sidebar.slider(
    "Hybrid Alpha",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)
st.sidebar.markdown("---")

show_evaluation = st.sidebar.checkbox(
    "Show Evaluation Metrics"
)
@st.cache_resource
def load_tfidf_model():

    vectorizer = joblib.load(
        "models/tfidf/tfidf_vectorizer.pkl"
    )

    tfidf_matrix = joblib.load(
        "models/tfidf/tfidf_matrix.pkl"
    )

    return vectorizer, tfidf_matrix


vectorizer, tfidf_matrix = load_tfidf_model()

df = pd.read_csv(
    "data/processed/quora_processed_1000.csv"
)
documents = df["processed_text"].tolist()

tokenized_documents = [
    doc.split()
    for doc in documents
]

bm25_model = build_bm25_model(
    tokenized_documents,
    k1=k1,
    b=b
)
query = st.text_input(
    "Enter Query"
)
refined_query = refine_query(
    query
)
method = st.selectbox(
    "Search Method",
    [
        "TF-IDF",
        "BM25",
        "Hybrid Parallel",
        "Hybrid Serial"
    ]
)

if st.button("Search"):

    if not query:

        st.warning(
            "Please enter a query"
        )
    
    else:

        st.success(
            f"Method: {method}"
        )
        st.info(
            f"Original Query: {query}"
        )

        st.info(
            f"Refined Query: {refined_query}"
        )
        
        if method == "TF-IDF":

            results, scores = search_tfidf(
                refined_query,
                vectorizer,
                tfidf_matrix,
                top_k=10
            )

            st.subheader("Results")

            for rank, idx in enumerate(
                results,
                start=1
            ):

                with st.container():

                    st.markdown(
                        f"### Rank #{rank}"
                    )

                    st.write(
                        f"Score: {round(float(scores[idx]),4)}"
                    )

                    st.write(
                        df.iloc[idx]["original_text"]
                    )

                    st.divider()

        elif method == "BM25":
    
            query_tokens = refined_query.split()

            results, scores = search_bm25(
                query_tokens,
                bm25_model,
                top_k=10
            )

            st.subheader("Results")

            for rank, idx in enumerate(
                results,
                start=1
            ):

                with st.container():

                    st.markdown(
                        f"### Rank #{rank}"
                    )

                    st.write(
                        f"Score: {round(float(scores[idx]),4)}"
                    )

                    st.write(
                        df.iloc[idx]["original_text"]
                    )

                    st.divider()

        elif method == "Hybrid Parallel":
    
            _, tfidf_scores = search_tfidf(
                refined_query,
                vectorizer,
                tfidf_matrix,
                top_k=len(documents)
            )

            query_tokens = refined_query.split()

            _, bm25_scores = search_bm25(
                query_tokens,
                bm25_model,
                top_k=len(documents)
            )

            results, scores = hybrid_search(
                tfidf_scores,
                bm25_scores,
                alpha=alpha,
                top_k=10
            )

            st.subheader("Results")

            for rank, idx in enumerate(
                results,
                start=1
            ):

                with st.container():

                    st.markdown(
                        f"### Rank #{rank}"
                    )

                    st.write(
                        f"Score: {round(float(scores[idx]),4)}"
                    )

                    st.write(
                        df.iloc[idx]["original_text"]
                    )

                    st.divider()

        elif method == "Hybrid Serial":
    
            _, tfidf_scores = search_tfidf(
                query,
                vectorizer,
                tfidf_matrix,
                top_k=len(documents)
            )

            query_tokens = query.split()

            _, bm25_scores = search_bm25(
                query_tokens,
                bm25_model,
                top_k=len(documents)
            )

            results = serial_hybrid_search(
                tfidf_scores,
                bm25_scores,
                candidate_size=50,
                top_k=10
            )

            st.subheader("Results")

            for rank, idx in enumerate(
                results,
                start=1
            ):

                with st.container():

                    st.markdown(
                        f"### Rank #{rank}"
                    )

                    st.write(
                        f"BM25 Score: {round(float(bm25_scores[idx]),4)}"
                    )

                    st.write(
                        df.iloc[idx]["original_text"]
                    )

                    st.divider()
        if show_evaluation:
    
            retrieved_docs = list(results)

            relevant_docs = retrieved_docs[:5]

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

            ap = average_precision(
                retrieved_docs,
                relevant_docs
            )

            ndcg = ndcg_at_k(
                retrieved_docs,
                relevant_docs,
                k=10
            )

            st.markdown("---")

            st.subheader(
                "Evaluation Metrics"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Precision@10",
                    round(precision, 4)
                )

                st.metric(
                    "Recall",
                    round(recall, 4)
                )

            with col2:

                st.metric(
                    "MAP",
                    round(ap, 4)
                )

                st.metric(
                    "nDCG",
                    round(ndcg, 4)
                )