from sentence_transformers import CrossEncoder


model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_results(query, results, top_k=3):

    pairs = []

    for result in results:
        pairs.append([
            query,
            result["chunk"]
        ])

    scores = model.predict(pairs)

    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])

    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]