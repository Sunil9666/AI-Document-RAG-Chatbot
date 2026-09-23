import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def create_bm25_index(chunks):

    tokenized_chunks = [
        tokenize(chunk)
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def keyword_search(bm25, chunks, query, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for index in ranked_indices:

        results.append({
            "index": index,
            "score": float(scores[index]),
            "chunk": chunks[index]
        })

    return results


def hybrid_search(
    faiss_index,
    bm25,
    chunks,
    query_embedding,
    query,
    top_k=5
):

    # -----------------------------
    # 1. Semantic Search
    # -----------------------------

    query_embedding = query_embedding.astype("float32")

    semantic_scores, semantic_indices = faiss_index.search(
        query_embedding,
        top_k
    )

    # -----------------------------
    # 2. Keyword Search
    # -----------------------------

    keyword_results = keyword_search(
        bm25,
        chunks,
        query,
        top_k
    )

    # -----------------------------
    # 3. Combine Results
    # -----------------------------

    combined = {}

    # Add semantic results
    for i, index in enumerate(semantic_indices[0]):

        combined[int(index)] = {
            "index": int(index),
            "semantic_score": float(
                semantic_scores[0][i]
            ),
            "keyword_score": 0.0
        }

    # Add keyword results
    for result in keyword_results:

        index = result["index"]

        if index not in combined:

            combined[index] = {
                "index": index,
                "semantic_score": 0.0,
                "keyword_score": 0.0
            }

        combined[index]["keyword_score"] = result["score"]

    # -----------------------------
    # 4. Normalize scores
    # -----------------------------

    results = list(combined.values())

    max_semantic = max(
        [r["semantic_score"] for r in results],
        default=1
    )

    max_keyword = max(
        [r["keyword_score"] for r in results],
        default=1
    )

    for result in results:

        semantic_normalized = (
            result["semantic_score"] / max_semantic
            if max_semantic != 0
            else 0
        )

        keyword_normalized = (
            result["keyword_score"] / max_keyword
            if max_keyword != 0
            else 0
        )

        # Give both methods equal importance
        result["combined_score"] = (
            0.5 * semantic_normalized
            +
            0.5 * keyword_normalized
        )

        result["chunk"] = chunks[
            result["index"]
        ]

    # -----------------------------
    # 5. Sort by combined score
    # -----------------------------

    results.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return results[:top_k]