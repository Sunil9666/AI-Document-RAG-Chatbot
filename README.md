# 🤖 AI Document RAG Chatbot

An AI-powered document question-answering system that allows users to upload PDF documents and ask questions based on the uploaded document.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from documents and generate answers using an LLM.

## 🚀 Features

- 📄 Upload PDF documents
- 💬 Ask questions about uploaded PDFs
- 🔎 Semantic document search
- ⚡ FAISS + BM25 hybrid search
- 🎯 Cross-Encoder reranking
- 🤖 LLM-powered answers
- 📚 Display relevant sources
- 💡 Dynamic suggested questions
- 🔄 New suggested questions after every chat
- 👀 PDF preview
- 📋 Copy AI responses
- ⚛️ React frontend
- 🐍 FastAPI backend

## 🛠️ Tech Stack

### Frontend
- React.js
- JavaScript
- HTML
- CSS

### Backend
- Python
- FastAPI
- Uvicorn

### AI / RAG
- LLM
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- Cross-Encoder
- FAISS
- BM25

## 🏗️ Architecture

```text
User
 ↓
React Frontend
 ↓
FastAPI Backend
 ↓
PDF Text Extraction
 ↓
Text Chunking
 ↓
Embeddings
 ↓
FAISS + BM25
 ↓
Hybrid Retrieval
 ↓
Cross-Encoder Reranking
 ↓
Relevant Context
 ↓
LLM
 ↓
Answer + Sources
 ↓
Suggested Questions
 ↓
React UI
