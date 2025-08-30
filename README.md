<!-- Decorative header image used in the README. 
     Replace the path below with the final repo path you intend to use 
     (e.g., ./assets/readme-header.png). -->
<p align="center">
  <img src="/mnt/data/b1ed897b-8ea7-402b-8cf5-acde81882746.png" alt="Decorative header image" width="100%" />
</p>

# Personal Propositional RAG Chatbot (Prototype)

> A local-first personal research assistant that ingests PDF/TXT/MD documents, extracts atomic propositions, stores them in a vector database, and answers questions with precise citations—running privately on your machine.

**Tech:** Python · Gradio · Pinecone · MIT License

---

## 🔍 Quick Reference

This prototype implements a local retrieval-augmented generation (RAG) system focused on propositions. It parses documents into paragraphs, generates atomic facts via an LLM “propositionizer,” embeds them using Sentence-Transformers, stores vectors in a Pinecone index, and delivers answers with context-rich citations. **Security:** only `.pdf`, `.txt`, and `.md` files up to **10 MB** are accepted.

### Key Features

- **📄 Document ingestion pipeline** — Parse PDFs/TXT/MD, extract propositions with an LLM, encode with Sentence-Transformers, and upsert into Pinecone.
- **💬 Chat interface** — Gradio 5.x single-page UI with SSR and PWA support.
- **🔍 Retrieval & answering** — Embed query, similarity search, top-k filtering, and generate an answer with citations.
- **💸 Cost awareness** — Real-time display of LLM usage and token counts.
- **📊 Evaluation harness** — Retrieval metrics: hit@k, nDCG, and citation span accuracy.
- **🧠 (Optional) MoE retrieval** — Router, gate, and two-stage reranker for hybrid retrieval.

---

## 🧰 Technologies

| Category            | Technology / Version Hint                    |
|---------------------|----------------------------------------------|
| Programming Language| Python ≥ 3.11                                |
| UI Framework        | Gradio ≥ 5.42.0                              |
| Vector Database     | Pinecone ≥ 7.0                               |
| Embeddings          | Sentence-Transformers 5.x                    |
| ML Runtime          | PyTorch ≥ 2.8                                |
| PDF Parser          | `pypdf` ≥ 6.0                                |
| LLM API             | OpenRouter `/chat/completions`               |

**Live demo:** The Gradio interface runs locally at `http://localhost:7860` after starting the app.

---

## 🗺 Visual Architecture Overview

### High-Level System Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[User documents (.pdf/.txt/.md)] --> B[Parsers (PDF/TXT/MD)]
        B --> C[Paragraphs]
        C --> D[Propositionizer (LLM)]
        D --> E[Atomic propositions]
        E --> F[Sentence-Transformers]
        F --> G[Embeddings]
        G --> H{Vector DB<br/>Pinecone index}
    end
    subgraph UI
        U[Gradio Chat Interface] -->|User query| V[Embedder]
        V -->|Query vector| H
        H -->|Top-k vectors| W[Context Composer]
        W --> X[LLM (OpenRouter)]
        X -->|Answer & citations| U
    end
    subgraph Optional MoE Retrieval
        R[Router] -->|Scores| G1[Gate]
        G1 -->|k-experts| Retrieval[Retrieve Candidates]
        Retrieval --> S[Reranker]
        S -->|Ranked results| W
    end
    H -->|Store embeddings & metadata| VectorStore[(Pinecone Index)]
