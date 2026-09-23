import faiss
import numpy as np


def create_vector_store(embeddings):

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    # Inner Product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_vector_store(index, query_embedding, top_k=5):

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    return distances, indices