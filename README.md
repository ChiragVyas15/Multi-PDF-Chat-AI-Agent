# Multi-PDF-Chat-AI-Agent
🤖 RAG-based Multi-PDF Chat Agent for semantic search and intelligent document QA without API keys.


### 📌 Short Project Description

**Multi-PDF Chat Agent** is a Retrieval-Augmented Generation (RAG) based application that allows users to upload multiple PDF documents and ask questions across them. The system extracts text, creates embeddings using Sentence Transformers, stores them in a FAISS vector database, and generates answers using Flan-T5, providing a completely free, API-key-free document question-answering solution. 

---

# README.md

```markdown
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

```

├── app.py
├── requirements.txt
├── faiss_index/
├── img/
└── README.md

````

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Multi-PDF-Chat-Agent.git
cd Multi-PDF-Chat-Agent
````

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Application

```bash
streamlit run app.py
```

## Example Queries

* Summarize the uploaded documents.
* What are the key findings?
* Explain a specific topic from the PDFs.
* Compare concepts across documents.

## Dependencies

* streamlit
* pypdf
* sentence-transformers
* faiss-cpu
* transformers
* torch
* numpy

## Future Improvements

* Conversation memory
* Support for DOCX and TXT files
* Chat history
* Multiple LLM options
* Source citation highlighting

## Author

**Chirag Vyas**

M.Tech AI & Data Science, IIIT Kota

## License

This project is licensed under the MIT License.

```

---

### 🔹 GitHub Repository Description (Short)

> **A free RAG-based Multi-PDF Chat Agent built with Streamlit, Sentence Transformers, FAISS, and Flan-T5 for intelligent document question answering without requiring API keys.**
```
