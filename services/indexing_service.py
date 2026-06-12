from collections import defaultdict


def build_inverted_index(df):

    inverted_index = defaultdict(list)

    for _, row in df.iterrows():

        doc_id = row["doc_id"]

        terms = row["processed_text"].split()

        for term in set(terms):

            inverted_index[term].append(doc_id)

    return inverted_index