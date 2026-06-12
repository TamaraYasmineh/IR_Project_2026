import re


def refine_query(query):

    query = query.lower()

    query = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        query
    )

    query = query.strip()

    corrections = {
        "pyhton": "python",
        "machin": "machine",
        "lern": "learn"
    }

    words = []

    for word in query.split():

        words.append(
            corrections.get(
                word,
                word
            )
        )

    return " ".join(words)