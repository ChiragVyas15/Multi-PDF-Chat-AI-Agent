"""
Multi-PDF Chat Agent — 100% Free Version
------------------------------------------
A Streamlit app that lets users upload multiple PDFs and ask questions
across all of them using Retrieval-Augmented Generation (RAG).

Replaces (all paid / API-key required):
    GoogleGenerativeAIEmbeddings (Gemini)  -> sentence-transformers (free, local)
    ChatGoogleGenerativeAI (gemini-pro)    -> google/flan-t5-large (free, local)
    GOOGLE_API_KEY                          -> not needed at all

Run with:
    streamlit run app.py
"""

import os

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Multi PDF Chatbot", page_icon=":scroll:")

FAISS_INDEX_PATH = "faiss_index"


# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models... (first run may take a minute)")
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return embedding_model, tokenizer, qa_model


embedding_model, qa_tokenizer, qa_model = load_models()


# ---------------------------------------------------------------------------
# Stage 1: Read PDFs
# ---------------------------------------------------------------------------
def get_pdf_text(pdf_docs) -> str:
    """Extract and concatenate text from multiple uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# ---------------------------------------------------------------------------
# Stage 2: Chunk text
# ---------------------------------------------------------------------------
def get_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Stage 3: Build & save the FAISS vector store
# ---------------------------------------------------------------------------
def get_vector_store(text_chunks: list[str]):
    """Embed chunks and build/save a FAISS index, persisted to disk."""
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))

    # Save the chunk text alongside the index (FAISS only stores vectors)
    with open(os.path.join(FAISS_INDEX_PATH, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk.replace("\n", " ") + "\n<<<CHUNK_END>>>\n")

    return index, text_chunks


def load_vector_store():
    """Load a previously saved FAISS index and its chunks from disk."""
    index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "index.faiss"))
    with open(os.path.join(FAISS_INDEX_PATH, "chunks.txt"), "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = [c.strip() for c in raw.split("<<<CHUNK_END>>>") if c.strip()]
    return index, chunks


# ---------------------------------------------------------------------------
# Stage 4: Retrieve relevant chunks
# ---------------------------------------------------------------------------
def retrieve_relevant_chunks(query: str, chunks: list[str], index, top_k: int = 4) -> list[str]:
    """Return the top_k chunks most similar to the query."""
    query_embedding = embedding_model.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ---------------------------------------------------------------------------
# Stage 5: Generate answer (replaces LangChain's load_qa_chain + Gemini)
# ---------------------------------------------------------------------------
def generate_answer(query: str, chunks: list[str], index, top_k: int = 4) -> tuple[str, list[str]]:
    """Run the full RAG pipeline: retrieve context, then generate an answer."""
    relevant_chunks = retrieve_relevant_chunks(query, chunks, index, top_k=top_k)
    context = "\n\n".join(relevant_chunks)

    prompt = f"""Answer the question as detailed as possible from the provided context,
make sure to provide all the details. If the answer is not in the provided
context, just say "answer is not available in the context". Don't provide
a wrong answer.

Context:
{context}

Question: {query}

Answer:"""

    inputs = qa_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = qa_model.generate(**inputs, max_length=300, num_beams=4, early_stopping=True)
    answer = qa_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return answer, relevant_chunks


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def user_input(user_question: str):
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        st.warning("Please upload and process PDF files first (use the sidebar).")
        return

    index, chunks = load_vector_store()

    with st.spinner("Thinking..."):
        answer, sources = generate_answer(user_question, chunks, index)

    st.write("Reply: ", answer)

    with st.expander("📚 Show source chunks used"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**Source {i}:**")
            st.write(src)
            st.divider()


def main():
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 (Free Version)")
    st.caption(
        "Powered by sentence-transformers + FAISS + Flan-T5-Large — "
        "no API key required."
    )

    user_question = st.text_input("Ask a Question from the PDF Files uploaded.. ✍️📝")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        if os.path.exists("img/Robot.jpg"):
            st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & Click on the Submit & Process Button",
            accept_multiple_files=True,
            type="pdf",
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                st.success(f"Done! Processed into {len(text_chunks)} chunks.")

        st.write("---")
        if os.path.exists("img/gkj.jpg"):
            st.image("img/gkj.jpg")
        st.write("AI App created by @ Gurpreet Kaur (Free version)")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/gurpreetkaurjethra" target="_blank">Gurpreet Kaur Jethra</a> | Made with ❤️ | Free version (no API key)
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
