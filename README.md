# 📚 Multi-Document RAG System

Query multiple PDF documents simultaneously using Retrieval-Augmented Generation (RAG). Upload any number of PDFs and ask questions that search across all documents at once.

## 🚀 Live Demo

Try it here: [multidoc-rag.streamlit.app](https://multidoc-rag.streamlit.app)

## ✨ Features

- **Multi-PDF Upload** — Upload as many documents as you want
- **Cross-Document Search** — Questions search ALL documents simultaneously
- **Source Attribution** — Every result shows which document it came from
- **Relevance Scoring** — Cosine similarity scores for every retrieved chunk
- **100% Free** — Uses Sentence Transformers (no API keys needed)
- **Source Breakdown** — Visual summary of which documents contributed results

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | Python |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | NumPy (cosine similarity) |
| PDF Processing | PyPDF2 |

## 📊 How It Works
