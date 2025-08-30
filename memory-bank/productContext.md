# Product Requirements Document (PRD)  
**Project:** Personal Propositional RAG Chatbot (Prototype)  
**Version:** 1.0.0  
**Last Updated:** August 29, 2025  

---

## 1. Vision & Purpose

**Goal:**  
Create a **local-first, personal research chatbot** that ingests PDF/TXT/MD documents, extracts **atomic propositions**, stores them in a vector database, and provides **precise answers with citations** to document spans.

**Why:**  
- Solve the "AI trust gap" in personal knowledge management.  
- Improve recall by surfacing evidence instead of opaque summaries.  
- Run privately on a personal machine (Windows PC), avoiding cloud lock-in.  

**Scope:**  
- **Prototype use only** (single user, non-production).  
- Must rely on **free / open-source tools** where possible.  
- Designed with a **path to production** (scalable, secure, MoE optional).  

---

## 2. User Stories

- **As a user**, I can upload documents (PDF, TXT, MD) so the system can parse them.  
- **As a user**, I can ask questions in a chat UI and get **answers with citations** pointing back to exact text spans.  
- **As a user**, I can monitor approximate **LLM costs** in real time.  
- **As a user**, I can evaluate retrieval quality (hit@k, nDCG, citation accuracy).  
- **As a power user**, I can enable **experimental MoE retrieval** to test hybrid/dense reranking pipelines.  

---

## 3. Core Features

1. **Document Ingestion Pipeline**  
   - Parse PDF/TXT/MD → paragraphs.  
   - LLM-based propositionizer extracts **atomic facts**.  
   - Sentence-Transformers 5.x encodes vectors (dense + sparse optional).  
   - Upsert into Pinecone 7.x index.  

2. **Chat Interface**  
   - Gradio 5.x UI (`gr.ChatInterface`) with SSR and PWA support.  
   - Mobile-optimized, local browser accessible at `http://localhost:7860`.  
   - Streaming optional (disabled by default for simplicity).  

3. **Retrieval & Answering**  
   - User query → embedded → similarity search.  
   - Retrieve top-k results, filter dynamically.  
   - Compose evidence context and call OpenRouter `/chat/completions`.  
   - Citations in format `[file:page:start-end]`.  

4. **Cost & Performance Awareness**  
   - Simple UI panel for **real-time API credit usage**.  
   - Display token count per query.  

5. **Evaluation Harness**  
   - Offline tests for retrieval metrics: hit@k, nDCG@k, citation span accuracy.  
   - Basic A/B testing harness (baseline vs MoE).  

6. **Optional MoE Retrieval (Experimental)**  
   - Expert router (centroids).  
   - Selective retrieval gate (adaptive k).  
   - Two-stage reranking (cross-encoder + optional LLM).  
   - Toggle via `MOE_ENABLED=true` in `.env`.  

---

## 4. Technical Requirements

### 4.1 Core Stack
- **Python**: 3.11+ (min 3.10).  
- **UI**: Gradio ≥5.42.0.  
- **Vector DB**: Pinecone ≥7.0.0 (`pinecone[grpc]`).  
- **Embeddings**: Sentence-Transformers ≥5.1.0.  
- **ML Runtime**: PyTorch ≥2.8.0.  
- **PDF Parser**: `pypdf` ≥6.0.0.  
- **LLM API**: OpenRouter `/chat/completions`.  

### 4.2 Local Deployment
- Run on Windows 10/11 with PowerShell 5.1+.  
- Minimum 4GB RAM; CPU-only with optional OpenVINO acceleration.  
- No Docker/Kubernetes required (but keep Dockerfile ready for future).  

### 4.3 Config & Env
- YAML config (`config.yaml`) plus `.env` for secrets.  
- Keys:  
  - `OPENROUTER_API_KEY`  
  - `PINECONE_API_KEY`  
- Safe defaults for all other params (max file size 10MB, top-k=6).  

---

## 5. Security & Privacy

- **Local-first**: all parsing/embedding done on device.  
- **External calls**: only to OpenRouter and Pinecone.  
- **File upload restrictions**: `.pdf`, `.txt`, `.md`, max 10MB.  
- **Model trust disabled**: `trust_remote_code=false`.  
- **API keys**: stored only in `.env`, not committed to repo.  

---

## 6. Performance Targets

- **UI startup**: <2s (Gradio 5.x SSR).  
- **Query latency**:  
  - Embedding (CPU): ≤5s / 100 sentences.  
  - Retrieval (Pinecone): ≤200ms.  
  - LLM call: user-dependent, but ≤10s typical.  
- **Memory**: <4GB baseline on CPU.  

---

## 7. Migration & Setup

### 7.1 Install (Prototype)
```bash
# Python env
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
