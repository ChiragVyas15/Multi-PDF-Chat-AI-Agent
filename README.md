# Multi-PDF-Chat-AI-Agent
🤖 RAG-based Multi-PDF Chat Agent for semantic search and intelligent document QA without API keys.


# 📚 Multi-PDF Chat Agent 🤖

A Retrieval-Augmented Generation (RAG) based Streamlit application that allows users to upload multiple PDF files and ask questions across all documents.

Unlike paid solutions, this project uses completely free and local models without requiring any API keys.

## 🚀 Features

- 📄 Upload multiple PDF documents
- ✂️ Automatic text chunking
- 🔍 Semantic search using FAISS
- 🧠 Sentence embeddings using all-MiniLM-L6-v2
- 🤖 Answer generation using Flan-T5-Large
- 💯 No API key required
- 🌐 Interactive Streamlit interface

## 🛠️ Technologies Used

- Python
- Streamlit
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- PyTorch
- NumPy
- PyPDF

## ⚙️ Working Pipeline

1. Upload PDF documents.
2. Extract text from PDFs.
3. Split text into chunks.
4. Generate embeddings using SentenceTransformer.
5. Store embeddings in FAISS vector database.
6. Retrieve relevant chunks based on user query.
7. Generate answers using Flan-T5-Large.

## 📂 Project Structure
