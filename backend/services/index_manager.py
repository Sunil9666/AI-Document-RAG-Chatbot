import os
import json
import pickle
import faiss


def save_index(index, bm25, chunks, output_dir="vectorstore"):
    os.makedirs(output_dir, exist_ok=True)

    # Save FAISS index
    faiss.write_index(
        index,
        os.path.join(output_dir, "index.faiss")
    )

    # Save BM25 index
    with open(
        os.path.join(output_dir, "bm25.pkl"),
        "wb"
    ) as f:
        pickle.dump(bm25, f)

    # Save chunks
    with open(
        os.path.join(output_dir, "chunks.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_index(output_dir="vectorstore"):

    # Load FAISS
    index = faiss.read_index(
        os.path.join(output_dir, "index.faiss")
    )

    # Load BM25
    with open(
        os.path.join(output_dir, "bm25.pkl"),
        "rb"
    ) as f:
        bm25 = pickle.load(f)

    # Load chunks
    with open(
        os.path.join(output_dir, "chunks.json"),
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    return index, bm25, chunks