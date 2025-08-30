# AGENTS.md: AI Collaboration Guide (Merged with MoE‑Style Retrieval)

<!-- This document serves as the authoritative guide for AI agents working within the `personal‑rag‑propo` repository. It conveys project goals, architecture, tool usage, safety constraints, operational runbooks, and test criteria. Agents must adhere to the instructions herein to ensure consistent behavior, high‑quality code, and secure operations. All guidance is derived from the repository contents and accompanying rule sets; each non‑trivial claim is grounded with a citation. External research about mixture‑of‑experts (MoE) retrieval is preserved as optional, research‑gated features and not independently re‑verified. -->

> **Verification Note:** External research claims about MoE‑style retrieval are preserved in this guide but have **not** been re‑verified. They are labeled in the External Research section (R#). All repository‑grounded statements cite internal sources using tether IDs such as [[1]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L4-L11). When uncertain, treat MoE features as experimental and optional.

## 1. Project Overview & Purpose

**Primary Goal.** Build a local‑first retrieval‑augmented chatbot that ingests personal PDF/TXT/MD files, extracts **atomic propositions** using a large language model, embeds them, stores them in a vector database, and answers questions with precise citations back to the document and character span [[1]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L4-L11) [[2]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L38-L56).

**Business Domain.** Personal knowledge management and research productivity. The application operates as a research assistant that surfaces evidence from user‑provided documents, improving recall and citation accuracy.

**Key Features.**

- **Ingestion pipeline:** Parses supported file types (PDF/TXT/MD) into paragraphs, calls an LLM to extract propositions, encodes them with a sentence‑transformer, and upserts vectors to a Pinecone index [[3]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/parsers.py#L5-L34) [[4]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L13-L46).
- **Chat interface:** Provides a Gradio multi‑tab UI (Chat, Ingest, Settings, Costs) built with `gr.Blocks` and `gr.ChatInterface` [[2]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L38-L56).
- **Retrieval & answering:** Embeds user queries, performs a top‑k similarity search in the vector store, composes a context string, and calls OpenRouter's `/chat/completions` endpoint to generate concise answers with citations [[5]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L21-L48).
- **Cost estimator:** Computes monthly LLM costs based on query volume and token usage via a dedicated tab in the UI [[6]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L63-L71).
- **Evaluation harness:** Provides Hit@k, nDCG@k, and span accuracy metrics to benchmark proposition‑level retrieval [[7]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/eval/eval.py#L7-L28).
- **Optional MoE‑style retrieval:** Incorporates a lightweight expert router, selective retrieval gate and reranker cascade using Pinecone metadata filters. Enable this feature only if evaluation gates are met and the improvements outweigh the added complexity; treat it as experimental.

## 2. Core Technologies & Stack

- **Languages:** Python 3.10+ for all backend code. The UI is defined in Python via Gradio.
- **Frameworks & Runtimes:**
  - **Gradio 5.44.1:** Provides the web UI, including chat, uploads, tabs and event handling [[2]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L38-L56). The application uses the `gr.Blocks` pattern recommended for complex apps.
  - **Pinecone Serverless:** Stores 384‑dimensional dense embeddings. The index is created with `ServerlessSpec` specifying cloud and region configuration [[8]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/config.py#L23-L34) [[9]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/vectorstore.py#L7-L17). Namespaces provide per‑user isolation and can be deleted to purge data.
  - **OpenRouter API:** LLM provider used via HTTP POST with Bearer authentication [[10]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L21-L38). Rate limits and credit usage must be monitored.
  - **Sentence‑Transformers v5.1.0:** Embeds propositions and queries using a BGE‑small (or similar) model. Models are cached to avoid repeated loading and run on CPU/GPU as available [[11]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/embeddings.py#L1-L8).
  - **Cross‑Encoder reranker (optional):** When MoE retrieval or dynamic re‑ranking is enabled, a lightweight cross‑encoder reorders retrieved hits. We recommend `cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2` because it balances ranking quality (NDCG@10 ≈ 74.30; MRR@10 ≈ 39.01) and throughput (≈1800 docs/sec) [[12]](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). For higher throughput with slightly lower accuracy, use `MiniLM‑L‑4‑v2` (NDCG@10 ≈ 73.04; 2500 docs/sec) or `MiniLM‑L‑2‑v2` (NDCG@10 ≈ 71.01; 4100 docs/sec).
- **Databases:** Pinecone vector database for semantic search; each tenant is isolated via namespaces [[4]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L13-L46).
- **Key Libraries:** `tqdm` for progress bars during ingestion [[13]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L15-L16), `pypdf` for PDF parsing [[14]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/parsers.py#L3-L17), `dotenv` for configuration, and `requests` for HTTP calls.
- **Package Manager:** `pip` with `requirements.txt`; environment uses a virtual environment created via `python -m venv` [[15]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/README.md#L11-L24).
- **Platforms:** Intended for local Windows/Linux execution; optionally deployable to cloud hosts. Uses Pinecone serverless service for the vector database.

## 3. Architectural Patterns & Structure

- **Overall Architecture:** A monolithic Python application orchestrating distinct agents (Ingest, Propositionizer, Retrieval, Evaluation and optional MoE modules) through modular functions. Ingestion, vector operations, LLM calls and UI concerns are separated [[16]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/README.md#L51-L59).
- **Directory Structure Philosophy:**
  - `/app.py` -- entry point launching the Gradio UI and wiring configuration, embedder and index initialization [[17]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L10-L18).
  - `/src/config.py` -- dataclass for environment variables [[8]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/config.py#L23-L34).
  - `/src/parsers.py` -- file reading and paragraph splitting [[3]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/parsers.py#L5-L34).
  - `/src/propositionizer.py` -- calls the LLM to extract propositions [[18]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L24-L46).
  - `/src/embeddings.py` -- model caching for sentence embeddings [[11]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/embeddings.py#L1-L8).
  - `/src/vectorstore.py` -- Pinecone operations including index creation and queries [[19]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/vectorstore.py#L7-L26).
  - `/src/ingest.py` -- end‑to‑end ingestion pipeline using parsers, propositionizer, embeddings and vectorstore [[4]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L13-L46).
  - `/src/rag.py` -- retrieval and answer composition logic [[5]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L21-L48).
  - `/src/eval/eval.py` -- evaluation metrics (Hit@k, nDCG@k, span accuracy) [[7]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/eval/eval.py#L7-L28).
  - Optional MoE modules: `router.py`, `gate.py` and `rerank.py` for expert routing, retrieval gating and re‑ranking.
  - `/data` -- sample text files [[20]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/README.md#L61-L62).
  - `/docs/reference_material` -- research sources and pricing benchmarks.
- **Module Organization:** Each module encapsulates a single responsibility. High‑level functions such as `ingest_files` and `rag_chat` orchestrate multiple steps but maintain pure function design where possible [[4]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L13-L46) [[5]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L21-L48).
- **Common Patterns & Idioms:**
  - **Functional design:** Agents are implemented as functions accepting configuration, models and data and returning structured outputs [[4]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L13-L46).
  - **Caching:** Embedding models are cached globally to avoid repeated initialization [[11]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/embeddings.py#L1-L8).
  - **Composition over inheritance:** Complex behavior (context composition, cost calculation) is built by composing smaller functions [[21]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L12-L19).
  - **Synchronous I/O:** The current code is synchronous; network calls are blocking [[5]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L21-L48). Concurrency may be added in the future with proper rate‑limit handling.

## 4. Coding Conventions & Style Guide

- **Formatting:** Follow PEP 8 with 4‑space indentation, 100‑character line limit and snake_case naming. Use `black` for automatic formatting. Avoid trailing whitespace and ensure a newline at EOF.
- **Naming Conventions:** Functions and variables use `snake_case` (`ingest_files`, `parent_excerpt`) [[4]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L13-L46); classes use `PascalCase` (e.g., `AppConfig`) [[8]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/config.py#L23-L34); constants use `UPPER_SNAKE_CASE` (e.g., `SYS_PROMPT`) [[1]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L4-L11).
- **API Design:** Keep functions pure where practical. Expose high‑level entry points rather than requiring callers to orchestrate internal steps. Use type hints and one‑line docstrings to describe input/output [[4]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L13-L46).
- **Error Handling:** Wrap external API calls in try/except blocks and implement exponential backoff for OpenRouter. Skip invalid propositions and continue processing [[22]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L25-L45). Do not hardcode secrets or API keys [[8]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/config.py#L23-L34).
- **Forbidden Patterns:** Avoid global mutable state except for model caches. Do not mutate chat history; conversation memory is currently stateless.

## 5. Development & Testing Workflow

- **Local Setup:**
  - Create and activate a virtual environment: `python -m venv .venv && .\.venv\Scripts\activate` on Windows [[23]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/README.md#L11-L18).
  - Install dependencies: `pip install -r requirements.txt` [[24]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/README.md#L18-L21) and optionally `pip install "pinecone[grpc]"`.
  - Copy `.env.example` to `.env` and set API keys and configuration values [[25]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/README.md#L32-L47).
  - Run the application: `python app.py`; Gradio prints a local URL [[26]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/README.md#L26-L30).
- **Testing:** Use `pytest` for unit tests with network calls mocked. Use `eval.py` to compute hit@k, nDCG@k and span accuracy on benchmark datasets [[7]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/eval/eval.py#L7-L28). Use a golden corpus to detect regressions.
- **Linting & Formatting:** Run `black` and `isort`. Use a linter like `flake8` or `ruff` to catch unused imports and style violations.
- **CI/CD:** Configure GitHub Actions to run tests and linting on pull requests. Consider adding nightly evaluation to monitor retrieval quality.

## 6. Git Workflow & PR Instructions

- **Pre‑Commit Checks:** Run tests and linters before committing. Validate ingestion and retrieval functions on sample data. Ensure no secrets are committed. Exclude `.env` from version control.
- **Branching Strategy:** Use feature branches off `main`. Each feature or bug fix should reside in its own branch (e.g., `feat/ingest-validation`, `fix/rate-limit-handling`). Avoid pushing directly to `main`.
- **Commit Messages:** Follow Conventional Commit format (e.g., `feat:`, `fix:`). Summarize what changed and why. Provide context for any breaking changes.

## 7. Security Considerations

- **Input Validation:** Accept only `.pdf`, `.txt` and `.md` uploads. Enforce file size limits (e.g., 10 MB). Sanitize file paths and reject malicious content.
- **Secrets Management:** Read keys from environment variables; never hardcode them [[8]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/config.py#L23-L34).
- **HTTP & LLM:** Use HTTPS for network calls. Handle HTTP errors and propagate user‑friendly messages. Implement retry with exponential backoff on 4xx/5xx status codes.
- **Least Privilege:** Use Pinecone namespaces for per‑user isolation and restrict API keys to specific indices. Avoid storing sensitive data in vector metadata; include only excerpts needed for retrieval.
- **Dependency Management:** Pin dependency versions and monitor security advisories.

## 8. Specific Agent Instructions & Known Issues

### 8.1 Agent Catalog

| Agent | Role & Description | Inputs | Outputs | Tools & Libs | Memory / State | Policies |
|-------|-------------------|--------|---------|--------------|----------------|----------|
| **IngestAgent** | Parse files, extract propositions, embed them and upsert to the vector store; shows a progress bar [[4]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L13-L46). | `files: List[Path]`, `namespace: str` | JSON report (counts, errors) | `pypdf`, propositionizer, sentence‑transformers, Pinecone, `tqdm` | Embedding model cache [[11]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/embeddings.py#L1-L8) | Validate file types and sizes; per‑user namespace; rate‑limit LLM calls [[18]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L24-L46) |
| **PropositionizerAgent** | Extracts self‑contained propositions (≤30 words) from each paragraph via LLM and returns JSON with `text`, `span` and tags [[1]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L4-L11) [[18]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L24-L46). | `paragraphs: List[str]` | List of propositions with parent excerpt | OpenRouter API | Stateless | Use deterministic temperature (0); validate JSON and skip invalid entries [[18]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L24-L46) |
| **RetrievalAgent** | Embeds question, queries vector store, composes context and calls LLM for answer [[5]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L21-L48). | `message: str`, `history: List[Tuple[str,str]]` | Answer string | Sentence‑transformers, Pinecone, OpenRouter | Stateless; model cache | Use `top_k` and similarity thresholds; include citations `[file:page:start-end]`; handle no‑match case gracefully [[27]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L41-L48) |
| **EvalAgent** | Computes Hit@k, nDCG@k and span accuracy to evaluate retrieval [[7]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/eval/eval.py#L7-L28). | `relevant_ids`, `predicted_ids`, `true_spans`, `pred_spans` | Metrics | Pure Python | Stateless | Return 0 for empty inputs; handle semicolon‑separated lists [[7]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/eval/eval.py#L7-L28) |
| **(MoE) ExpertRouterAgent** | Routes queries to top‑N experts by cosine similarity. Returns chosen expert IDs and similarity scores. | `q_emb`, `expert_centroids: Dict[str, ndarray]` | `chosen_experts: List[str]`, `sims: Dict[str, float]` | `numpy` | Centroids refreshed periodically | Default `top_n=1–2`; fallback to all experts if confidence is low (research‑gated). |
| **(MoE) SelectiveGateAgent** | Decides whether to retrieve and chooses adaptive `k` based on similarity thresholds. | `q_emb`, `router_sims`, `cfg` | `do_retrieve: bool`, `k: int` | Heuristic threshold logic | Stateless | Configurable thresholds for retrieval and k; conservative defaults. |
| **(MoE) RerankerStage1Agent** | Re‑ranks hits using a small cross‑encoder; stage‑1 of MoE cascade. | `query`, `hits` | `reranked_hits` | Cross‑encoder (e.g., MiniLM) | Stateless | Always on when MoE enabled; reduces candidate set for stage‑2. |
| **(MoE) RerankerStage2Agent** | Optional LLM re‑ranks uncertain results; stage‑2 of MoE cascade. | `query`, `reranked_hits_s1` | `reranked_hits_final` | OpenRouter LLM | Stateless | Invoke only when uncertainty exceeds a threshold; expensive. |

### 8.2 Tooling & Connectors

- **Gradio (UI):** Use `gr.Blocks` and `gr.ChatInterface`. Keep component definitions in a context manager and assign explicit labels/defaults. Use `gr.State()` to store temporary chat memory if future enhancements require it.
- **Pinecone:** Initialize the client with an environment key; create the index with `ServerlessSpec` and dimension matching the embedding model [[9]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/vectorstore.py#L7-L17). Upsert vectors with metadata (doc_id, file name, text, parent excerpt, span, tags). Use namespaces for isolation.
- **OpenRouter API:** Authenticate using a Bearer token. Send messages with `temperature=0` and handle HTTP errors with retry/backoff.
- **Sentence‑Transformers:** Load models once and cache them in a global dictionary [[11]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/embeddings.py#L1-L8). Use `encode(batch_size=...)` and normalize embeddings before similarity comparisons.
- **Cross‑Encoder Reranker:** Load the chosen cross‑encoder (e.g., `MiniLM‑L‑6‑v2`). Compute scores for `(query, passage)` pairs and re‑order hits based on these scores [[12]](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
- **Misc Libraries:** `tqdm` for progress display, `python‑dotenv` for config, `pypdf` for PDF parsing.

### 8.3 Grounding & Retrieval Pipeline

1. **Parse & Paragraph:** Call `parse_any(path)` to detect file type and read PDF/TXT/MD files. Split the text into paragraphs via `to_paragraphs()`, merging short segments [[3]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/parsers.py#L5-L34).

2. **Proposition Extraction:** For each paragraph, call `propositionize_paragraphs()` on the PropositionizerAgent. The system prompt instructs the LLM to extract self‑contained propositions, each ≤30 words, returning JSON with `propositions` including `text` and `span` information [[1]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L4-L11). The function posts to OpenRouter, parses the JSON and adds the parent excerpt if missing [[18]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L24-L46).

3. **Embed & Upsert:** Each proposition's text is embedded using the sentence‑transformer model with normalization [[28]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L28-L38). A unique vector ID is constructed from the document hash and a truncated SHA1 of the proposition [[29]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L17-L32). Vectors are batched and upserted to Pinecone along with metadata [[30]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py#L30-L41). Ensure the index exists by calling `ensure_index()` before ingestion [[9]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/vectorstore.py#L7-L17).

4. **(MoE) Route & Gate:** If MoE retrieval is enabled, compute the query embedding (`q_emb`), route to the top experts via cosine similarity and decide whether to retrieve using the selective gate. The gate chooses `k` between `k_min` and `k_max` based on similarity thresholds.

5. **Query & Compose:** Embed the user question, query the vector store for the top `k` neighbors (optionally filtered by `expert_id`) and include metadata. Compose the context by formatting each match as `[file:page:start-end] text -- parent_excerpt` [[31]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L14-L19), preserving order and separating entries by blank lines. If no matches are returned, respond with "No strong match found." [[32]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L45-L48).

6. **(MoE) Rerank Cascade:** Apply a cross‑encoder reranker to reorder the retrieved hits. If the uncertainty of the result exceeds a threshold (`uncertainty_threshold`), invoke a second LLM reranker for further refinement. Otherwise, skip stage‑2 to save latency and cost.

7. **LLM Answering:** Pass a system prompt, the composed context and the user question to OpenRouter via `/chat/completions` with `temperature=0` [[33]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L21-L36). Parse the response and return the answer.

8. **Evaluation:** Compute Hit@k, nDCG@k and span accuracy using functions from `eval.py` [[7]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/eval/eval.py#L7-L28) to benchmark retrieval quality.

9. **Dynamic Retrieval & Re‑Ranking (Advanced):** To improve precision and control context length, enable adaptive retrieval and re‑ranking:
   - **Score thresholds:** Filter results by cosine similarity using configurable cutoffs. A similarity threshold near 0.5 yields broad results, while thresholds in the 0.7--0.8 range yield stricter matches. We recommend `low_score_cutoff=0.5` and `high_score_cutoff=0.8` so that high‑confidence queries return fewer passages and low‑confidence queries return more.
   - **Adaptive `k`:** Define `min_top_k`, `max_top_k` and `default_top_k` (e.g., 4, 15 and 8). Query up to `max_top_k` candidates and then choose `k`: if the top similarity ≥ `high_score_cutoff`, set `k=min_top_k`; if ≤ `low_score_cutoff`, set `k=max_top_k`; otherwise use `default_top_k`.
   - **Cross‑encoder reranker:** After retrieving candidates, apply a lightweight cross‑encoder to reorder hits. The recommended model `ms‑marco‑MiniLM‑L‑6‑v2` balances accuracy and speed (NDCG@10 ≈ 74.30, 1800 docs/sec) [[12]](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). Consider `MiniLM‑L‑4‑v2` (≈73.04; 2500 docs/sec) or `MiniLM‑L‑2‑v2` (≈71.01; 4100 docs/sec) when resources are limited. Use the cross‑encoder scores to re‑order hits before composing the context.

### 8.4 Memory & State Management

- **Short‑Term Memory:** The chat pathway is currently stateless; the `history` parameter is ignored [[36]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L41-L49). Future enhancements can use `gr.State()` to store recent turns and include a summary in the retrieval step.
- **Long‑Term Memory:** All propositions are stored in the Pinecone vector store under a namespace. Namespaces provide isolation and allow deletion. Be aware of eventual consistency and refresh the index after ingestion when necessary.
- **Cache:** Embedding models are cached in a global dictionary to avoid repeated loading [[11]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/embeddings.py#L1-L8). When changing the embedding model, invalidate the cache and recreate the index with the new dimension.
- **Invalidation:** Use Pinecone's delete API (`index.delete(ids=...)` or `delete_all=True`) to remove obsolete vectors. Restart the application to reload configuration changes.

### 8.5 Safety, Compliance & Observability

- **Rate Limits & Cost Guards:** Enforce a 0.2 second delay between LLM calls during propositionization to stay within free tier limits [[18]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L24-L46). Monitor OpenRouter credit usage via the `/key` endpoint and set per‑key credit limits. Batch upserts to reduce write units.
- **RBAC & Data Isolation:** Use per‑user namespaces and restrict API keys to specific indices. Do not store sensitive personal information in vector metadata; only store excerpts necessary for retrieval.
- **Logging & Telemetry:** Instrument ingestion and query functions with structured logs. Log vector counts, ingestion duration, query latency, error events and retrieval scores. Propagate request IDs for correlation. Do not log user content or full LLM responses.
- **Compliance:** Use HTTPS for all network calls. Do not expose the app publicly without authentication. For enterprise deployments, implement user authentication and audit logging. Comply with data retention policies and provide a mechanism for users to delete their data.

### 8.6 Operations (Runbook, Failure Modes & Fallbacks)

- **Startup:** Ensure `.env` contains valid API keys. Run `python app.py`. On first run, the embedding model downloads and the index is ensured [[17]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L10-L18).
- **Ingestion:** In the **Ingest** tab, upload supported files, specify a namespace and click **Ingest**. Monitor progress via the progress bar and review the JSON report on completion [[37]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L51-L55). If ingestion fails, review logs for file type or parsing errors.
- **Chat:** In the **Chat** tab, ask questions about ingested documents. Answers include citations `[file:page:start-end]`. If no matches are found, the system responds with a fallback message [[32]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py#L45-L48).
- **Cost Estimation:** Use the **Costs** tab to adjust parameters (monthly query count, prompt tokens, completion tokens, price per 1K tokens and base fixed cost) and compute the estimated monthly cost [[38]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L21-L30) [[6]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py#L63-L71).
- **Failure Modes & Fallbacks:**
  - **LLM Errors:** Retry HTTP errors from OpenRouter and display a friendly message. For a 402 (insufficient credits), prompt the user to recharge.
  - **Vector Store Issues:** If the index does not exist or dimensions mismatch, call `ensure_index()` again or recreate the index [[9]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/vectorstore.py#L7-L17).
  - **Ingestion Aborts:** Large documents may take time; progress bars help. Partial upserts may leave stale vectors---delete the namespace and re‑ingest.
  - **File Validation:** Reject unsupported file types or oversized files with a clear error message.
  - **Consistency Lag:** Newly ingested vectors may not be immediately searchable due to eventual consistency. Suggest waiting or performing a dummy query.
  - **(MoE) Router Failure:** If expert routing has low confidence, fall back to retrieving across all experts (no filter). If the LLM reranker is overused, raise the uncertainty threshold or set a per‑query cost cap.

### 8.7 Acceptance Criteria & Test Harness

- **Ingestion Verification:** Ingesting sample documents returns a report with non‑zero proposition counts and all vectors upserted. Calling `index.describe_index_stats()` confirms vector counts. Unsupported file suffixes must raise a `ValueError` [[39]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/parsers.py#L35-L43).
- **Retrieval Evaluation:** For a benchmark Q/A set with ground truth proposition IDs and spans, compute Hit@k, nDCG@k and span accuracy [[7]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/eval/eval.py#L7-L28). Use these metrics to compare proposition‑level retrieval against passage‑level retrieval. Do not claim improvements without evidence.
- **UI Functionality:** All Gradio tabs load correctly. Uploading files triggers ingestion; asking questions triggers retrieval; the Costs tab computes an estimate without raising exceptions. Input validation prevents unsupported file types and invalid numbers.
- **Error Handling:** Simulate network failures; the app should catch exceptions and return descriptive messages. Invalid JSON responses from the propositionizer should result in skipping those propositions [[18]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L24-L46).
- **MoE Adoption Gates:** If enabling MoE retrieval, evaluate on a sample of 50--200 Q/A pairs. Adopt MoE only if nDCG@k improves ≥ 3%, span accuracy improves ≥ 2%, latency increase ≤ 25% and per‑query cost remains within budget. Otherwise, disable MoE in configuration.

### 8.8 Changelog

| Date | Author | Description |
|------|--------|-------------|
| 2025‑08‑29 | Lead Orchestrator | Initial creation of `AGENTS.md`. Documented project purpose, architecture, agent catalogue, RAG pipeline, safety, operations, acceptance criteria and changelog, with citations from code and rulesets. |
| 2025‑08‑29 | Lead Orchestrator | Merged MoE‑style retrieval: added router/gate/reranker agents, Pinecone metadata filtering, configuration flags, evaluation gates and risks. External research claims labeled R#. |
| 2025‑08‑30 | Lead Orchestrator | Added dynamic retrieval thresholds (`low_score_cutoff=0.5`, `high_score_cutoff=0.8`, adaptive `min_top_k=4`, `default_top_k=8`, `max_top_k=15`). Recommended cross‑encoder rerankers and documented their performance [[12]](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). Updated configuration and pipeline accordingly. |

## 9. Configuration (MoE Flags & Retrieval Thresholds)

Store these values in a YAML or environment‑backed configuration (loaded by `AppConfig` in `config.py`) [[8]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/config.py#L23-L34). Example:

```yaml
router:
  enabled: true
  experts: ["general", "code", "personal_notes"]
gate:
  enabled: true
  retrieve_sim_threshold: 0.62      # route below this similarity triggers retrieval
  low_sim: 0.45                     # if router max similarity < low_sim, use k_max
retrieval:
  high_score_cutoff: 0.8            # high similarity threshold for dynamic retrieval
  low_score_cutoff: 0.5             # low similarity threshold for dynamic retrieval
  default_top_k: 8                  # fallback number of results when scores are moderate
  k_min: 4                          # minimum number of results to return
  k_max: 15                         # maximum number of results to consider
reranker:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  alternatives:
    - cross-encoder/ms-marco-MiniLM-L-4-v2
    - cross-encoder/ms-marco-MiniLM-L-2-v2
  stage2_enabled: true
  uncertainty_threshold: 0.15       # triggers expensive LLM reranker when uncertainty is high
```

## 10. Risks & Reversions (MoE)

- **Domain drift and misrouting:** Expert centroids may become stale; refresh periodically. If router confidence is below `retrieve_sim_threshold`, fall back to retrieving across all experts. Monitor similarity distributions and adjust thresholds accordingly.
- **Latency creep:** Stage‑2 reranker increases latency; adjust `uncertainty_threshold` upward or disable stage‑2 to stay within budget.
- **Over‑filtering:** Narrow metadata filters may exclude relevant passages; allow a union of the top‑N experts and occasionally query without filters if recall is low.
- **Metadata integrity:** Validate `expert_id` on ingest and normalize labels.

## 11. Appendix --- Minimal Code Inserts (MoE)

Below are illustrative diffs for integrating MoE retrieval into the current codebase.

**config.yaml**

```yaml
router:
  enabled: true
  experts: ["general", "code", "personal_notes"]
gate:
  enabled: true
  retrieve_sim_threshold: 0.62
  low_sim: 0.45
retrieval:
  high_score_cutoff: 0.8
  low_score_cutoff: 0.5
  default_top_k: 8
  k_min: 4
  k_max: 15
reranker:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  alternatives: [cross-encoder/ms-marco-MiniLM-L-4-v2, cross-encoder/ms-marco-MiniLM-L-2-v2]
  stage2_enabled: true
  uncertainty_threshold: 0.15
```

**router.py**

```python
import numpy as np

def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-12)

def route_query_to_experts(q_emb: np.ndarray, expert_centroids: dict, top_k: int = 1):
    """Return top‑k expert IDs and similarity scores."""
    q_norm = _normalize(q_emb)
    sims = {eid: float(np.dot(q_norm, _normalize(c))) for eid, c in expert_centroids.items()}
    chosen = sorted(sims, key=sims.get, reverse=True)[:top_k]
    return chosen, sims
```

**gate.py**

```python
def retrieval_gate(router_sims: dict, cfg) -> tuple[bool, int]:
    """Decide whether to retrieve and choose k based on similarity thresholds."""
    max_sim = max(router_sims.values()) if router_sims else 0.0
    do_retrieve = max_sim < cfg.router.retrieve_sim_threshold
    # adapt k based on similarity thresholds
    if not do_retrieve:
        return False, cfg.retrieval.k_min
    if max_sim < cfg.router.low_sim:
        return True, cfg.retrieval.k_max
    return True, cfg.retrieval.default_top_k
```

**rag.py** (simplified query path)

```python
# embed the query
q_emb = embedder.encode([query], normalize_embeddings=True)[0]
# route and gate (MoE)
chosen, sims = route_query_to_experts(q_emb, expert_centroids, top_k=2)
do_retrieve, k = retrieval_gate(sims, cfg)
if not do_retrieve:
    return no_context_response(query)
# query vector store with dynamic k and optional expert filter
hits = index.query(
    vector=q_emb.tolist(),
    top_k=k,
    filter={"expert_id": {"$in": chosen}},
    include_metadata=True,
)
# dynamic retrieval thresholds
candidates = hits["matches"]
# apply cross‑encoder reranker
pairs = [(query, m["metadata"]["text"]) for m in candidates]
rerank_scores = cross_encoder.predict(pairs)
reranked = [m for m, _ in sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)]
context = compose_context(reranked)
# optionally stage‑2 reranking (not shown)
```

## References

Repository‑grounded citations are embedded throughout the document in brackets, using tether IDs such as [[1]](https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py#L4-L11) to reference specific lines in the source files (see `src/propositionizer.py`, `app.py`, etc.). External research citations (R#) are listed separately below; they inform optional MoE‑style retrieval features but have not been independently validated.

**External Research (R#)**

- R1 -- *Clustered Adaptive Mixture of Experts (CAME)*: arXiv Nov 2023.
- R2 -- *RouterRetriever: Expert Routing over Embedding Models*: arXiv Sep 2024; HTML Feb 2025.
- R3 -- *Self‑RAG (Retrieve on Demand)*: arXiv Oct 2023; OpenReview 2024.
- R4 -- *Understanding the Design Decisions of RAG*: arXiv Nov 2024.
- R5 -- *Sufficient Context & Guided Abstention*: arXiv Nov 2024.
- R6 -- *Reranker Benchmarks (LLM vs. Light Models)*: arXiv Aug 2025.
- R7 -- Pinecone metadata filtering docs & serverless design paper: docs, Jul 2025.

---

**End of document.**
