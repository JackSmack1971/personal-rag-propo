# AGENTS.md: AI Collaboration Guide

<!-- This document serves as the authoritative guide for AI agents working within the personal-rag-propo repository. It conveys project goals, architecture, tool usage, safety constraints, operational runbooks, and test criteria. Agents must adhere to the instructions herein to ensure consistent behavior, high-quality code, and secure operations. All guidance is derived from the repository contents and accompanying rule sets; each non-trivial claim is grounded with a citation. -->

## 1. Project Overview & Purpose

**Primary Goal:** Build a local-first retrieval-augmented chatbot that ingests personal PDF/TXT/MD files, extracts atomic propositions using a large language model, embeds them, stores them in a vector database, and answers questions with precise citations back to the document and character span [1][2].

**Business Domain:** Personal knowledge management and research productivity. The application operates as a research assistant that surfaces evidence from user-provided documents, improving recall and citation accuracy.

**Key Features:**

- **Ingestion pipeline:** Parses supported file types (PDF/TXT/MD) into paragraphs, calls an LLM to extract propositions, encodes them with a sentence-transformer, and upserts vectors to a Pinecone index [3][4].
- **Chat interface:** Provides a Gradio multi-tab UI (Chat, Ingest, Settings, Costs) built with `gr.Blocks` and `gr.ChatInterface` [2].
- **Retrieval & answering:** Embeds user queries, performs a top-k similarity search in the vector store, composes a context string, and calls OpenRouter’s `/chat/completions` endpoint to generate concise answers with citations [5].
- **Cost estimator:** Computes monthly LLM costs based on query volume and token usage via a dedicated tab [6].
- **Evaluation harness:** Provides Hit@k, nDCG@k, and span accuracy metrics to benchmark proposition-level retrieval [7].

## 2. Core Technologies & Stack

**Languages:** Python 3.10+ for all backend code. The UI is defined using Python via Gradio.

**Frameworks & Runtimes:**

- **Gradio 5.44.1:** Provides the web UI, including chat, file uploads, tabs, and event handling [2]. Follows the Blocks pattern recommended for complex apps 【gradio-5-44-1-ruleset.md#Application Architecture and Initialization】.
- **Pinecone Serverless:** Stores 384-dimensional dense vector embeddings; the index is created with `ServerlessSpec` specifying cloud and region [8][9].
- **OpenRouter API:** Serves as the LLM provider; endpoints are called via HTTP POST with Bearer token authentication [10] 【openrouter-api-ruleset.md#Authentication and API Keys】.
- **Sentence-Transformers v5.1.0:** Embeds propositions and queries using a BGE-small model; models are cached to avoid repeated instantiation [11].
- **Databases:** Pinecone vector database used for semantic search; each tenant (namespace) isolates user data [4].
- **Key Libraries/Dependencies:** `tqdm` for progress bars during ingestion [12], `pypdf` for PDF parsing [13], `dotenv` for configuration, and `requests` for HTTP calls.
- **Package Manager:** `pip` with `requirements.txt`; environment uses a virtual environment created via `python -m venv` [14].
- **Platforms:** Intended to run on local Windows/Linux machines with optional future deployment to cloud hosts; uses Pinecone’s serverless service.

## 3. Architectural Patterns & Structure

**Overall Architecture:** Monolithic Python application orchestrating distinct agents (Ingest, Propositionizer, Retrieval, Evaluation) through modular functions. There is a clear separation between data ingestion, vector operations, LLM interactions, and UI layers [15].

**Directory Structure Philosophy:**

- **/app.py:** Entry point launching the Gradio UI and wiring configuration, embedder, and index initialization [16].
- **/src:** Core modules: `config.py` (dataclass for environment variables) [8], `parsers.py` (file reading & paragraph splitting) [3], `propositionizer.py` (calls LLM to extract propositions) [17], `embeddings.py` (model caching) [11], `vectorstore.py` (Pinecone operations) [18], `ingest.py` (end-to-end ingestion pipeline) [4], `rag.py` (retrieval and answer composition) [5], `eval/eval.py` (metrics) [7].
- **/data:** Sample text for demonstration [19].
- **/docs/reference_material:** Research sources and pricing benchmarks.

**Module Organization:** Each module encapsulates a single responsibility. Cross-module calls occur via clearly defined functions (e.g., `ingest_files`, `rag_chat`) rather than shared state. The configuration is loaded once and passed explicitly to each function [20].

**Common Patterns & Idioms:**

- **Functional design:** Agents are implemented as functions that accept configuration, models, and data and return structured outputs [4].
- **Caching & memoization:** The embedding model is cached in a global dictionary to avoid repeated loading [21].
- **Composition over inheritance:** Complex behaviors (e.g., context composition, cost calculation) are built by composing smaller functions rather than subclassing [22].
- **Async avoidance:** The current codebase is synchronous; all network calls are blocking and sequential [5]. Consider adding concurrency in the future using proper rate-limit handling.

## 4. Coding Conventions & Style Guide

**Formatting:** Follow PEP 8 with 4-space indentation, 100-character line limit, and `snake_case` naming for variables and functions. Use Black or equivalent autoformatters. Avoid trailing whitespace and ensure newline at EOF.

**Naming Conventions:**

- **Functions and variables:** `snake_case` (e.g., `ingest_files`, `parent_excerpt`) [4].
- **Classes:** `PascalCase` (e.g., `AppConfig`) [23].
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `SYS_PROMPT`) [1].
- **Files and modules:** `snake_case` (`vectorstore.py`, `rag.py`).

**API Design Principles:** Keep functions pure when possible, avoid side effects, and return structured data (dicts or dataclasses). Expose high-level entry points (`ingest_files`, `rag_chat`) instead of requiring consumers to orchestrate internal steps [4][5].

**Documentation Style:** Provide a one-line docstring describing purpose and input/outputs. Use type hints consistently for function parameters and return types, as seen in existing modules [4].

**Error Handling:** Wrap external API calls (OpenRouter/Pinecone) in `try/except` blocks, raising informative exceptions. On ingestion, skip invalid propositions and continue processing [24]. For network calls, implement exponential backoff per OpenRouter guidelines 【openrouter-api-ruleset.md#Error Handling and Response Processing】.

**Forbidden Patterns:** Never hardcode API keys or secrets; they must come from environment variables [8] 【openrouter-api-ruleset.md#Authentication and API Keys】. Do not store large objects in global variables except for model caching. Avoid mutating history outside `ChatInterface`; conversation memory is not yet supported.

## 5. Development & Testing Workflow

**Local Development Setup:**

1. Create and activate a virtual environment:  
   - Windows: `python -m venv .venv && .\.venv\Scripts\activate` [25]
2. Install dependencies:  
   - `pip install -r requirements.txt` [26].  
     Ensure `pinecone` is installed with gRPC extras (`pip install "pinecone[grpc]"`) to access serverless features 【pinecone_serverless_ruleset.md#Client Version Requirements】.
3. Copy `.env.example` to `.env` and set API keys and configuration values [27].
4. Launch the app with `python app.py`; Gradio will print a local URL [28].

**Build Commands:** There is no separate build step; the Python code runs directly. For front-end bundling or packaging, create Docker images or zipped deployments as needed.

**Testing Commands:**

- Use `pytest` to run unit tests (tests folder not yet included). Create tests for ingestion, retrieval, and cost modules. All network calls should be mocked; do not hit external APIs in tests 【AGENTS_md_STANDARD.md#5. Development & Testing Workflow】.
- Evaluate the retrieval pipeline with the provided `src/eval/eval.py` metrics. Prepare a benchmark dataset of questions and expected proposition IDs and spans, then compute hit@k, nDCG@k, and span accuracy [7].

**Linting/Formatting Commands:** Run `black` and `isort` on the codebase. Use `flake8` or `ruff` to catch unused imports and style issues. Ensure all code passes before committing.

**CI/CD Process Overview:** On pull request, run tests and linting in a GitHub Actions workflow. Block merges if any step fails. Consider adding a nightly evaluation job to monitor retrieval quality over time.

## 6. Git Workflow & PR Instructions

- **Pre-Commit Checks:** Before committing, run all tests and linters. Validate that ingestion and retrieval functions operate correctly on sample data and that no secrets are leaked. Ensure `.env` is not added to version control.
- **Branching Strategy:** Use feature branches off `main`. Each new feature or bug fix should live in its own branch (e.g., `feat/ingest-validation`, `fix/rate-limit-handling`). Avoid committing directly to `main`.
- **Commit Messages:** Follow Conventional Commit format (`feat:`, `fix:`, `docs:`). Summarize what changed and why. Provide context for any breaking changes.
- **Pull Request Process:** Keep PRs small and focused. Document the purpose, relevant files, and any test results. Ensure all CI checks pass. Request review from maintainers. Avoid `git push --force` on shared branches; use `--force-with-lease` only on your own branch when re-writing history.
- **Clean State:** Do not leave untracked or temporary files. Remove large artifacts (models, data) from the repo; rely on external storage or git-ignored caches.

## 7. Security Considerations

- **General Security Practices:** Always think like a security engineer. Validate all user inputs and sanitize file uploads to avoid injection attacks 【gradio-5-44-1-ruleset.md#Security and Production Considerations】. Only accept `.pdf`, `.txt`, or `.md` files and impose size limits (e.g., 10 MB) 【gradio-5-44-1-ruleset.md#Security and Production Considerations】.
- **Sensitive Data Handling:** Store OpenRouter and Pinecone API keys in environment variables, not in code or logs [8] 【openrouter-api-ruleset.md#Authentication and API Keys】. Do not log full requests or secrets.
- **Input Validation:** Use allowlists for file types; check file size before processing 【gradio-5-44-1-ruleset.md#Security and Production Considerations】. Validate numeric inputs in the cost estimator and guard against invalid floats [29].
- **Vulnerability Avoidance:** Avoid remote code execution by never executing user-supplied content. Use parameterized queries when adding metadata filters to Pinecone. Implement proper error handling for HTTP requests and display generic error messages to the user.
- **Dependency Management:** Pin versions in `requirements.txt` and monitor CVEs. Use `pip-tools` or Dependabot to update packages. Keep Pinecone and Sentence-Transformers versions aligned with the rulesets to maintain compatibility.
- **Principle of Least Privilege:** Limit API keys to the minimum scope. Use per-namespace isolation in Pinecone; never query across namespaces 【pinecone_serverless_ruleset.md#Namespace Strategy】. When running in production, restrict network egress and set up firewall rules.

## 8. Specific Agent Instructions & Known Issues

### 8.1 Agent Catalog

| Agent                | Role & Description                                                                 | Inputs                            | Outputs                                  | Tools                                                        | Memory/State                   | Policies                                                                                 |
|---------------------|-------------------------------------------------------------------------------------|-----------------------------------|-------------------------------------------|--------------------------------------------------------------|-------------------------------|------------------------------------------------------------------------------------------|
| IngestAgent         | File parsing, paragraph segmentation, proposition extraction, embedding & vector upsert; progress bar [4] | `files: List[Path]`, `namespace: str` | JSON report with file and proposition counts [30] | `pypdf`, propositionizer, sentence-transformers, pinecone, `tqdm` | Embedder cache; no persistent state [21] | File validation; per-user namespace; rate-limit LLM calls [17]                           |
| PropositionizerAgent| Extract self-contained propositions (≤30 words) via LLM [1]                          | `paragraphs: List[str]`           | Propositions list and parent excerpt [17] | OpenRouter API, JSON parsing                                 | Stateless; 0.2 s delay to respect rate limits [17] | No hallucination; dynamic model selection; handle JSON errors 【openrouter-api-ruleset.md#Model Selection and Validation】 |
| RetrievalAgent (RagChatAgent) | Embed question, query vectors, compose context & call LLM; fallback on no match [5] | `message: str`, `history: List[Tuple[str,str]]` | Answer string | Sentence-transformers, Pinecone, OpenRouter | Stateless; embedder cache; no chat history | Use `top_k`; context format `[doc:page:start-end]`; temperature 0; handle HTTP errors [31] 【openrouter-api-ruleset.md#Error Handling and Response Processing】 |
| EvalAgent           | Calculate Hit@k, nDCG@k & span accuracy [7]                                         | `relevant_ids: str`, `predicted_ids: str`, `true_spans: List[Dict]`, `pred_spans: List[Dict]` | Hit@k, nDCG@k, span accuracy | Pure Python | Stateless | Use semicolon-separated IDs; handle empty sets; return 0 when appropriate [7]           |

### 8.2 Tooling & Connectors

- **Gradio (UI):** Use `gr.Blocks` for multi-tab interfaces and call it within a context manager 【gradio-5-44-1-ruleset.md#Application Architecture and Initialization】. Use `gr.ChatInterface` for chat interactions 【gradio-5-44-1-ruleset.md#Interface vs Blocks Selection】. Always set explicit labels and default values for components; do not create components outside the Blocks context 【gradio-5-44-1-ruleset.md#Component Initialization Best Practices】. **[Grounded:** `gradio-5-44-1-ruleset.md#Application Architecture and Initialization`]**
- **Pinecone:** Initialize the client with an API key from environment variables. When creating an index, use `ServerlessSpec(cloud=<cloud>, region=<region>)` and ensure the dimension matches the embedding model [9] 【pinecone_serverless_ruleset.md#Serverless Index Creation】. Upsert vectors in batches and include metadata for document ID, file name, text, parent excerpt, span, and tags [4] 【pinecone_serverless_ruleset.md#Vector Upserts】. Use namespaces to isolate user data 【pinecone_serverless_ruleset.md#Namespace Strategy】. **[Grounded:** `pinecone_serverless_ruleset.md#Serverless Index Creation`]**
- **OpenRouter API:** Authenticate using a Bearer token and pass `HTTP-Referer` and `X-Title` headers if ranking is desired [32] 【openrouter-api-ruleset.md#Authentication and API Keys】. Always fetch available models dynamically and avoid hardcoding model IDs 【openrouter-api-ruleset.md#Model Selection and Validation】. Implement comprehensive error handling: check HTTP status, parse the `error` field, and retry on 408/429/502/503 errors with exponential backoff 【openrouter-api-ruleset.md#Error Handling and Response Processing】. **[Grounded:** `openrouter-api-ruleset.md#Error Handling and Response Processing`]**
- **Sentence-Transformers:** Cache models in a global dictionary to avoid repeated initialization [21]. Specify device (`cuda`/`cpu`) if necessary and use `encode()` with `batch_size` for throughput 【sentence_transformers_5_1_0_ruleset.md#Model Types and Core Usage Patterns】. Normalize embeddings for cosine similarity during both ingestion and query [33]. **[Grounded:** `sentence_transformers_5_1_0_ruleset.md#Model Types and Core Usage Patterns`]**
- **TQDM:** Use progress bars to provide user feedback during ingestion; wrap file iteration with `tqdm(files, desc="Ingest")` and update counts inside loops [34]. Always use a context manager or explicit `close()` to avoid resource leaks 【tqdm-4671-ruleset.md#Context Management and Resource Handling】. Configure `mininterval` and `miniters` when processing large datasets to improve performance 【tqdm-4671-ruleset.md#Performance Optimization】. **[Grounded:** `tqdm-4671-ruleset.md#Context Management and Resource Handling`]**
- **dotenv & Configuration:** Use `load_dotenv()` to load environment variables at startup. Define a dataclass (`AppConfig`) with all required configuration values and provide sensible defaults for development [8]. Expose a `from_env()` class method to populate the config [35]. **[Grounded:** `openrouter-api-ruleset.md#Authentication and API Keys`]**

### 8.3 Grounding & Retrieval Pipeline

- **Parse & Paragraph:** Use `parse_any(path)` to read PDF/TXT/MD files; it delegates to `read_pdf`, `read_text_file`, or `read_markdown` based on the suffix [36]. The function returns raw text and a list of paragraphs produced by `to_paragraphs()`, which merges short paragraphs to meet a minimum length [37].
- **Proposition Extraction:** For each paragraph, call `propositionize_paragraphs()` on the PropositionizerAgent. The system prompt instructs the model to extract self-contained propositions, each ≤30 words, returning JSON containing propositions, each with `text`, `span`, and optional `tags` [1]. The function loops through paragraphs, posts to OpenRouter, parses the JSON from the response, and appends the parent excerpt if missing [17].
- **Embed & Upsert:** Each proposition’s `text` is embedded using the Sentence-Transformer model with normalization [38]. A unique vector ID is constructed from the document hash and a truncated SHA1 of the proposition [39]. Vectors are batched and upserted to Pinecone via `index.upsert()` along with metadata (`doc_id`, `file` name, `text`, `parent_excerpt`, `span`, `tags`) [40]. Ensure the index exists by calling `ensure_index()` before ingestion; this checks for existence and creates the index if absent [9].
- **Query & Compose:** At query time, embed the user question with the same model and normalization [41]. Call `query()` on the vector store with `top_k=cfg.TOP_K` and `include_metadata=True` [42]. Compose the context by formatting each match as `[file:page:start-end] text -- parent_excerpt` [31], preserving ordering and separating entries with blank lines. If no matches are returned, immediately respond with “No strong match found.” [43].
- **LLM Answering:** Pass the system prompt (precise research assistant instructions) and composed context to OpenRouter via `/chat/completions`, with `temperature 0` to maximize determinism [44]. Use the `messages` array with `system` and `user` roles. Check HTTP status and parse the `choices[0].message.content` for the answer [45].
- **Evaluation:** Compute Hit@k (binary success), nDCG@k (graded rank quality), and span accuracy (intersection over union of character spans) using functions in `eval.py` [7]. Use these metrics to compare passage-level vs. proposition-level retrieval.

### 8.4 Memory & State Management

- **Short-Term Memory:** There is no persistent conversational state; each chat request is stateless. The `ChatInterface` passes history, but the current implementation ignores it [46]. Future enhancements could store recent turns in `gr.State()` and include them in retrieval with proper prompt management 【gradio-5-44-1-ruleset.md#Component State Management】.
- **Long-Term Memory:** All propositions are stored in the Pinecone vector store under a namespace. Namespaces provide tenant isolation and allow deletion of data per user 【pinecone_serverless_ruleset.md#Namespace Strategy】. Data is eventually consistent; newly upserted vectors may not immediately appear in queries 【pinecone_serverless_ruleset.md#Data Freshness】. Monitor ingestion completion and optionally refresh the index before querying.
- **Cache:** The Sentence-Transformer models are cached in `_model_cache` (dictionary) to prevent repeated loading [21]. The embedding model exposes `get_sentence_embedding_dimension()` to set the index dimension [47]. When updating the embedding model, invalidate the cache and recreate the Pinecone index with the new dimension.
- **Invalidation:** To remove obsolete or incorrect vectors, use the Pinecone delete API (`index.delete(ids=[...])` or `delete_all=True` per namespace) 【pinecone_serverless_ruleset.md#Vector Management】. For cost and evaluation configuration changes, restart the application to reload environment variables.

### 8.5 Safety, Compliance & Observability

- **Rate Limits & Cost Guards:** Enforce a 0.2 second sleep between LLM calls during propositionization to respect free tier limits [17]. Monitor OpenRouter credit usage via `/key` endpoint and set per-key credit limits 【openrouter-api-ruleset.md#Rate Limits and Credit Management】. Use Pinecone batch upserts to reduce write unit consumption 【pinecone_serverless_ruleset.md#Write Performance】.
- **RBAC & Data Isolation:** Use a separate Pinecone namespace per user (e.g., `namespace=f"user-{user_id}"`) to isolate data 【pinecone_serverless_ruleset.md#Namespace Strategy】. Restrict API keys to specific indices and scopes. Avoid storing sensitive data (personal information) in vector metadata; only store excerpts necessary for retrieval.
- **Logging & Telemetry:** Instrument ingestion and query functions with structured logging. Log vector counts, ingestion duration, query latency, and error events. Use logging levels (`info`, `warning`, `error`) and propagate request IDs for correlation. Do not log full OpenRouter responses or user data. For progress bars, rely on `tqdm` which handles cleanup on exceptions 【tqdm-4671-ruleset.md#Context Management and Resource Handling】.
- **Compliance:** Adhere to platform terms of service. Use HTTPS for all network calls. Do not expose the app on a public URL without authentication. For enterprise deployments, implement user authentication and audit logging. Comply with data retention policies and allow users to delete ingested data.

### 8.6 Operations (Runbook, Failure Modes & Fallbacks)

- **Startup:** Ensure `.env` is configured with OpenRouter and Pinecone keys. Run `python app.py`. On startup, the app loads the embedding model, ensures the index exists, and launches the Gradio server [20]. Expect a small delay while the model downloads on first run.
- **Ingestion Procedure:** In the **Ingest** tab, upload supported documents. Enter an optional namespace (defaults to `default`). Click **Ingest** to trigger the pipeline. Monitor progress via the progress bar and review the JSON report displayed on completion [48]. If ingestion fails for a file, check logs for exceptions (file type, size, parsing errors). Retry after fixing issues.
- **Chat Procedure:** In the **Chat** tab, ask questions about ingested documents. The model returns answers citing `[file:page:start-end]`. If no matches are found, the system responds with a fallback message [43].
- **Cost Estimation:** In the **Costs** tab, adjust monthly query count, prompt tokens, completion tokens, price per 1K tokens, and base fixed cost. Click **Estimate** to compute expected monthly cost [49][6].

**Failure Modes & Fallbacks:**

- **LLM Errors:** Handle HTTP errors from OpenRouter with retries and user-friendly messages. If a 402 error (insufficient credits) occurs, notify the user to recharge credits 【openrouter-api-ruleset.md#Error Handling and Response Processing】.
- **Vector Store Issues:** If the index does not exist, call `ensure_index()` again. For dimension mismatch errors, recreate the index with the correct dimension [9].
- **Ingestion Abort:** For large documents, ingestion may take time; use progress bars. If interrupted, the partial upsert may result in incomplete data—consider deleting the namespace and re-ingesting.
- **File Validation:** Reject unsupported file types or sizes with a clear error message 【gradio-5-44-1-ruleset.md#Security and Production Considerations】.
- **Eventual Consistency:** After ingestion, new vectors might not appear immediately. Suggest waiting a few seconds before querying or performing a dummy query to warm the index 【pinecone_serverless_ruleset.md#Data Freshness】.

### 8.7 Acceptance Criteria & Test Harness

- **Ingestion Verification:** Given a set of test documents, running `ingest_files()` must return a report with non-zero proposition counts and all vectors upserted. Use `index.describe_index_stats()` to verify vector counts. Files with unsupported suffixes must raise a `ValueError` [36].
- **Retrieval Evaluation:** For a benchmark set of Q/A pairs with ground truth proposition IDs and spans, compute hit@k, nDCG@k, and span accuracy using functions in `eval.py` [7]. Use these metrics to compare proposition-level retrieval against passage-level retrieval; do not claim improvements without evidence. **[Unverified]**
- **UI Functionality:** All Gradio tabs load correctly. Uploading files triggers ingestion; asking questions triggers retrieval; the costs tab calculates an estimate without raising exceptions. Input validation prevents unsupported file types and invalid numbers.
- **Error Handling:** Simulate network failures to OpenRouter and Pinecone; the application should not crash and should display an error message. Invalid JSON responses from the propositionizer should be handled by returning an empty propositions list [50].

### 8.8 Changelog

**Date (Y-M-D)** | **Author** | **Description**  
---|---|---  
**2025-08-29** | Lead Orchestrator | Initial creation of AGENTS.md. Documented the project purpose, architecture, agent catalogue, tooling guidelines, RAG pipeline, memory management, safety considerations, operations runbook, acceptance criteria, and changelog. Grounded all statements with citations from code and rulesets.

---

## References

[1] [17] [32] [50] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/propositionizer.py

[2] [6] [16] [20] [29] [47] [48] [49] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/app.py

[3] [13] [36] [37] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/parsers.py

[4] [12] [24] [30] [34] [38] [39] [40] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/ingest.py

[5] [10] [22] [31] [33] [41] [42] [43] [44] [45] [46] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/rag.py

[7] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/eval/eval.py

[8] [23] [35] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/config.py

[9] [18] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/vectorstore.py

[11] [21] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/src/embeddings.py

[14] [15] [19] [25] [26] [27] [28] GitHub  
https://github.com/JackSmack1971/personal-rag-propo/blob/main/README.md
