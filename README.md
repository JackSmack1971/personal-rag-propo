# Personal RAG Chatbot (Propositional RAG) — 2025 Stack

Local-first Windows-friendly starter with **Mixture of Experts (MoE) Architecture** using:
- **Gradio 5.x** with SSR for enhanced UI performance
- **Pinecone** with gRPC client for improved vector operations
- **OpenRouter** as the LLM provider with enhanced cost monitoring
- **Sentence-Transformers 5.x** with multi-backend support (torch/onnx/openvino)
- **Mixture of Experts (MoE)** with intelligent routing, adaptive retrieval, and multi-stage reranking

> This is a **personal** project scaffold featuring state-of-the-art 2025 technology stack. It favors free/OSS components and keeps a thin path to future deployment.

## Quickstart (Windows)

```bat
:: 1) Create & activate venv
python -m venv .venv
.\.venv\Scripts\activate

:: 2) Install deps
pip install -U pip
pip install -r requirements.txt

:: 3) Copy env template and edit keys
copy .env.example .env
notepad .env

:: 4) Launch app
python app.py
```

OpenGradio will print a `http://127.0.0.1:7860` link in the console.

## Configure

Set the following in `.env`:

```ini
# Core API Keys
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL=openrouter/auto
OPENROUTER_REFERER=http://localhost:7860
OPENROUTER_TITLE=Personal RAG (Propositional)
PINECONE_API_KEY=pcn-...
PINECONE_INDEX=personal-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Enhanced Configuration
EMBED_MODEL=BAAI/bge-small-en-v1.5
NAMESPACE=default
SENTENCE_TRANSFORMERS_BACKEND=torch  # torch/onnx/openvino

# MoE Configuration (Optional)
MOE_ENABLED=false  # Set to true to enable Mixture of Experts
MOE_ROUTER_ENABLED=true
MOE_GATE_ENABLED=true
MOE_RERANKER_ENABLED=true
```

> The embedding model is **384-d**, so the Pinecone index must be created with `dimension=384`.
> **MoE Features**: Set `MOE_ENABLED=true` to activate intelligent expert routing, adaptive retrieval, and multi-stage reranking for enhanced accuracy.

## What’s Inside

### Core Application
- `app.py` — launches the Gradio 5.x UI with SSR (tabs: **Chat**, **Ingest**, **Configuration**, **Cost Calculator**).
- `src/parsers.py` — PDF/TXT/MD parsing and paragraphing with enhanced security validation.
- `src/propositionizer.py` — LLM-based extraction of **atomic propositions** (JSON via OpenRouter).
- `src/embeddings.py` — Sentence-Transformers 5.x with multi-backend support (torch/onnx/openvino).
- `src/vectorstore.py` — Pinecone gRPC client for improved performance and reliability.
- `src/ingest.py` — end-to-end ingest: parse → propositionize → embed → upsert with progress tracking.
- `src/rag.py` — Enhanced RAG with optional MoE integration for intelligent retrieval.
- `src/config.py` — Enhanced configuration with MoE support and validation.

### Mixture of Experts (MoE) Architecture
- `src/moe/` — Complete MoE implementation directory:
  - `config.py` — MoE configuration management with YAML support
  - `router.py` — Expert routing with centroid management and performance tracking
  - `gate.py` — Selective retrieval gate with adaptive k-selection
  - `reranker.py` — Two-stage reranking (cross-encoder + LLM)
  - `integration.py` — Orchestration layer integrating all MoE components
- `scripts/validate_moe.py` — MoE integration validation and testing

### Evaluation & Metrics
- `src/eval/eval.py` — Advanced metrics: **hit@k**, **nDCG@k**, **citation span accuracy**.
- `src/eval/metrics.py` — Comprehensive retrieval metrics and performance tracking.
- `src/eval/ab_testing.py` — A/B testing framework for MoE vs baseline comparison.

### Documentation & Specifications
- `docs/specifications/` — Complete specification suite for all features
- `docs/research/` — Research reports and implementation recommendations
- `memory-bank/` — Project memory and decision documentation

### Sample Data & Scripts
- `data/sample/` — Sample documents for testing and demonstration
- `scripts/` — Utility scripts for validation, testing, and maintenance

## Mixture of Experts (MoE) Features

The system includes a complete **Mixture of Experts** architecture that enhances retrieval accuracy through intelligent routing and multi-stage processing:

### 🚀 Expert Routing
- **Intelligent Query Analysis**: Routes queries to the most appropriate specialized experts
- **Performance Learning**: Tracks expert performance and adjusts routing decisions
- **Centroid Management**: Maintains expert profiles based on document characteristics

### 🎯 Adaptive Retrieval Gate
- **Query Complexity Analysis**: Assesses query complexity to determine retrieval strategy
- **Dynamic K-Selection**: Adapts the number of retrieved documents based on confidence
- **Score Filtering**: Applies intelligent filtering to improve result quality

### 🔄 Two-Stage Reranking
- **Cross-Encoder Reranking**: Uses advanced models for precise relevance scoring
- **LLM-Based Reranking**: Optional second stage for complex reasoning tasks
- **Uncertainty Detection**: Automatically determines when advanced reranking is needed

### ⚙️ Configuration
Enable MoE features by setting `MOE_ENABLED=true` in your `.env` file:

```bash
# Enable MoE
MOE_ENABLED=true

# Fine-tune components (optional)
MOE_ROUTER_ENABLED=true
MOE_GATE_ENABLED=true
MOE_RERANKER_ENABLED=true
```

### 🧪 Validation
Test your MoE integration with the provided validation script:

```bash
python scripts/validate_moe.py
```

This will verify that all MoE components are properly integrated and functioning.

# Personal RAG Chatbot (Propositional) — Starter (COST defaults wired)

The **Costs** tab now reads defaults from `.env` populated with values derived from your spreadsheet:

- COST_MONTHLY_QS = 6000  (Medium scenario → Queries/day × 30)
- COST_PROMPT_TOKENS = 300
- COST_COMPLETION_TOKENS = 300
- COST_PRICE_PER_1K = 0.000375
- COST_BASE_FIXED = 50.0

You can change these in `.env` at any time
