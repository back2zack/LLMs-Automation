# LLMs-Automation
# 🔧 LLM Automation & RAG Workflows Playground

Welcome to my personal lab for **LLM-powered workflows**, **data preparation pipelines**, and **automation experiments** — built for real-world data engineering scenarios.

This repo is a hands-on collection of tools and prototypes focused on:
- 🧠 **Retrieval-Augmented Generation (RAG)** using custom document loaders & local LLMs
- ⚙️ **Automated document processing** (PDF → chunks → embeddings → vector DB)
- 💬 **API-based chatbot interfaces** (local/private models via Streamlit + REST)
- 📦 **Rapid experimentation** with LangChain, Ollama, and vector databases like Chroma

---

## 🚀 What You’ll Find Here

| Script | Purpose |
|--------|---------|
| `pdf_rag.py` | End-to-end RAG pipeline: PDF ingestion → chunking → embedding → multi-query retrieval → question answering |
`olla_Jung.py` | Streamlit-based chatbot that interacts with a local LLM persona ("Jung") via streaming REST API — designed for concise, psychology-inspired mentorship |


> These tools showcase my ability to integrate **LLMs into automated pipelines**, using modern Python frameworks and embedding services for scalable knowledge extraction.

---

## 👨‍💻 Why This Matters

This repo is part of my ongoing journey to master:
- LLM infrastructure and workflow automation
- Scalable vector search for enterprise-level data
- Data pipelines that bridge **raw unstructured data** and **semantic understanding**

Whether you’re building a **chatbot over internal documents**, automating **text ingestion**, or preparing datasets for **LLM fine-tuning**, this repo shows how I approach real-world data and NLP challenges.

---

## 🛠 Stack Highlights

- **LangChain** for building composable pipelines
- **Ollama** for local LLM inference (LLaMA, Mistral, etc.)
- **ChromaDB** for efficient vector storage and retrieval
- **Streamlit** for fast frontends
- **Dotenv + Requests** for clean API interactions

---

## 🌱 Future Ideas (WIP)

- Plug-and-play architecture for different models (local + cloud)
- Integration with cloud storage (S3/GCS) for large-scale document ingestion
- Automatic scheduling for batch embedding & RAG updates


