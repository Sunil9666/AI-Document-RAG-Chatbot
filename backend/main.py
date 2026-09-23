from fastapi import FastAPI, HTTPException

from backend.services.document_loader import extract_text_from_pdf
from backend.services.text_splitter import split_text
from backend.services.embeddings import create_embeddings
from backend.services.vector_store import create_vector_store
from backend.services.vector_store import (
    create_vector_store,
    search_vector_store
)
from backend.services.llm import (
    generate_answer,
    generate_suggested_questions
)
from backend.services.hybrid_search import (
    create_bm25_index,
    keyword_search
)
from backend.services.hybrid_search import (
    create_bm25_index,
    hybrid_search
)
from backend.services.reranker import rerank_results
from backend.services.index_manager import save_index, load_index

app = FastAPI(
    title="AI Document RAG Chatbot",
    description="AI-powered document search and question answering system",
    version="1.0.0"
)
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "AI Document RAG Chatbot is running!"
    }


@app.get("/test-pdf")
def test_pdf():

    file_path = "backend/uploads/employee_handbook.pdf"

    text = extract_text_from_pdf(file_path)

    return {
        "text": text
    }


@app.get("/test-chunks")
def test_chunks():

    file_path = "backend/uploads/project report.pdf"

    text = extract_text_from_pdf(file_path)

    chunks = split_text(text)

    return {
        "total_chunks": len(chunks),
        "chunks": chunks
    }

@app.get("/test-embeddings")
def test_embeddings():

    file_path = "backend/uploads/project report.pdf"

    text = extract_text_from_pdf(file_path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    return {
        "total_chunks": len(chunks),
        "embedding_shape": list(embeddings.shape)
    }

@app.get("/test-vector-store")
def test_vector_store():

    file_path = "backend/uploads/project report.pdf"

    text = extract_text_from_pdf(file_path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    index = create_vector_store(embeddings)

    return {
        "total_chunks": len(chunks),
        "vector_dimension": embeddings.shape[1],
        "vectors_stored": index.ntotal
    }


@app.get("/search")
def search_documents(query: str):

    file_path = "backend/uploads/project report.pdf"

    # 1. Extract PDF text
    text = extract_text_from_pdf(file_path)

    # 2. Split text into chunks
    chunks = split_text(text)

    # 3. Create embeddings for chunks
    embeddings = create_embeddings(chunks)

    # 4. Create FAISS index
    index = create_vector_store(embeddings)

    # 5. Convert user's question into an embedding
    query_embedding = create_embeddings([query])

    # 6. Search FAISS
    distances, indices = search_vector_store(
        index,
        query_embedding,
        top_k=5
    )

    # 7. Get relevant chunks
    results = []

    for i, index_value in enumerate(indices[0]):

        results.append({
            "chunk": chunks[index_value],
            "distance": float(distances[0][i])
        })

    return {
        "query": query,
        "results": results
    }

@app.get("/ask")
def ask_question(query: str):

    # Load active PDF index
    index, bm25, chunks = load_index("vectorstore")

    # Create query embedding
    query_embedding = create_embeddings([query])

    # Hybrid search
    results = hybrid_search(
        index,
        bm25,
        chunks,
        query_embedding,
        query,
        top_k=10
    )

    # Rerank results
    results = rerank_results(
        query,
        results,
        top_k=5
    )

    # Get relevant chunks
    relevant_chunks = [
        result["chunk"]
        for result in results
    ]

    # Create context
    context = "\n\n".join(
        relevant_chunks
    )

    # Generate answer
    answer = generate_answer(
        query,
        context
    )

    # Generate dynamic questions
    suggested_questions = generate_suggested_questions(
        query,
        answer,
        context
    )

    # DEBUG
    print("\n==============================")
    print("Suggested Questions:")
    print(suggested_questions)
    print("==============================\n")

    # IMPORTANT:
    # Send suggested_questions to React
    return {
        "question": query,
        "answer": answer,
        "sources": relevant_chunks,
        "suggested_questions": suggested_questions
    }

@app.get("/test-similarity")
def test_similarity(query: str):

    file_path = "backend/uploads/project report.pdf"

    text = extract_text_from_pdf(file_path)

    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    query_embedding = create_embeddings([query])

    index = create_vector_store(embeddings)

    distances, indices = search_vector_store(
        index,
        query_embedding,
        top_k=5
    )

    results = []

    for i, index_value in enumerate(indices[0]):

        results.append({
            "rank": i + 1,
            "similarity": float(distances[0][i]),
            "chunk": chunks[index_value]
        })

    return {
        "query": query,
        "results": results
    }

@app.get("/test-keyword-search")
def test_keyword_search(query: str):

    file_path = "backend/uploads/project report.pdf"

    text = extract_text_from_pdf(file_path)

    chunks = split_text(text)

    bm25 = create_bm25_index(chunks)

    results = keyword_search(
        bm25,
        chunks,
        query,
        top_k=5
    )

    return {
        "query": query,
        "results": results
    }
@app.get("/test-hybrid-search")
def test_hybrid_search(query: str):

    file_path = "backend/uploads/project report.pdf"

    # 1. Extract text
    text = extract_text_from_pdf(file_path)

    # 2. Create chunks
    chunks = split_text(text)

    # 3. Create embeddings
    embeddings = create_embeddings(chunks)

    # 4. Create FAISS
    index = create_vector_store(embeddings)

    # 5. Create BM25
    bm25 = create_bm25_index(chunks)

    # 6. Create query embedding
    query_embedding = create_embeddings([query])

    # 7. Hybrid search
    results = hybrid_search(
        index,
        bm25,
        chunks,
        query_embedding,
        query,
        top_k=5
    )

    return {
        "query": query,
        "results": results
    }

@app.get("/test-reranker")
def test_reranker(query: str):

    file_path = "backend/uploads/project report.pdf"

    # Extract text
    text = extract_text_from_pdf(file_path)

    # Create chunks
    chunks = split_text(text)

    # Create embeddings
    embeddings = create_embeddings(chunks)

    # FAISS
    index = create_vector_store(embeddings)

    # BM25
    bm25 = create_bm25_index(chunks)

    # Query embedding
    query_embedding = create_embeddings([query])

    # Hybrid retrieval
    results = hybrid_search(
        index,
        bm25,
        chunks,
        query_embedding,
        query,
        top_k=10
    )

    # Rerank the candidates
    results = rerank_results(
        query,
        results,
        top_k=5
    )

    return {
        "query": query,
        "results": results
    }
@app.get("/build-index")
def build_index():

    file_path = "backend/uploads/project report.pdf"

    # 1. Extract PDF text
    text = extract_text_from_pdf(file_path)

    # 2. Split into chunks
    chunks = split_text(text)

    # 3. Create embeddings
    embeddings = create_embeddings(chunks)

    # 4. Create FAISS index
    index = create_vector_store(embeddings)

    # 5. Create BM25 index
    bm25 = create_bm25_index(chunks)

    # 6. Save everything
    save_index(
        index,
        bm25,
        chunks,
        "vectorstore"
    )

    return {
        "message": "Index created and saved successfully",
        "total_chunks": len(chunks)
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    # Save uploaded PDF
    file_path = f"backend/uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extract text from the uploaded PDF
    text = extract_text_from_pdf(file_path)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this PDF."
        )

    # Split the uploaded PDF into chunks
    chunks = split_text(text)

    # Create embeddings for THIS PDF
    embeddings = create_embeddings(chunks)

    # Create FAISS index for THIS PDF
    index = create_vector_store(embeddings)

    # Create BM25 index for THIS PDF
    bm25 = create_bm25_index(chunks)

    # Replace the previous active index
    save_index(
        index,
        bm25,
        chunks,
        "vectorstore"
    )

    return {
        "message": "PDF uploaded and indexed successfully",
        "filename": file.filename,
        "total_chunks": len(chunks)
    }