# Personal RAG Chatbot (Propositional RAG) — Starter

Local-first Windows-friendly starter using:
- **Gradio** for the UI
- **Pinecone** as the vector database
- **OpenRouter** as the LLM provider
- **BAAI/bge-small-en-v1.5** (384-d) for embeddings

> This is a **personal** project scaffold. It favors free/OSS components and keeps a thin path to future deployment.

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
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL=openrouter/auto
OPENROUTER_REFERER=http://localhost:7860
OPENROUTER_TITLE=Personal RAG (Propositional)
PINECONE_API_KEY=pcn-...
PINECONE_INDEX=personal-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBED_MODEL=BAAI/bge-small-en-v1.5
NAMESPACE=default
```

> The embedding model is **384-d**, so the Pinecone index must be created with `dimension=384`.

## What’s Inside

- `app.py` — launches the Gradio UI (tabs: **Chat**, **Ingest**, **Settings**, **Costs**).
- `src/parsers.py` — PDF/TXT/MD parsing and paragraphing.
- `src/propositionizer.py` — LLM-based extraction of **atomic propositions** (JSON via OpenRouter).
- `src/embeddings.py` — loads **BGE-small-en-v1.5** and encodes text.
- `src/vectorstore.py` — Pinecone helpers (create/check index, upsert, query).
- `src/ingest.py` — end-to-end ingest: parse → propositionize → embed → upsert.
- `src/rag.py` — retrieval + context expansion + prompt composition + answer.
- `src/eval/eval.py` — simple metrics: **hit@k**, **nDCG@k**, **citation span accuracy**.
- `data/sample/` — tiny seed text.

# Personal RAG Chatbot (Propositional) — Starter (COST defaults wired)

The **Costs** tab now reads defaults from `.env` populated with values derived from your spreadsheet:

- COST_MONTHLY_QS = 6000  (Medium scenario → Queries/day × 30)
- COST_PROMPT_TOKENS = 300
- COST_COMPLETION_TOKENS = 300
- COST_PRICE_PER_1K = 0.000375
- COST_BASE_FIXED = 50.0

You can change these in `.env` at any time
