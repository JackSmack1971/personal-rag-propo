import os, json, requests
import numpy as np
from typing import List, Tuple, Optional
from .vectorstore import query as vector_query
from .security import get_secure_api_headers, log_security_event

# MoE Integration imports
from .moe.integration import get_moe_pipeline, process_query_with_moe, MoEPipelineResult
from .moe.config import get_moe_config

SYSTEM_PROMPT = (
    "You are a precise research assistant.\n"
    "Use ONLY the provided sources.\n"
    "Cite evidence as [doc:page:start-end]. If no sufficient evidence, reply: 'No strong match found'.\n"
    "Be concise.\n"
)

ENHANCED_SYSTEM_PROMPT = (
    "You are a precise research assistant with access to proposition-level document fragments.\n"
    "Use ONLY the provided sources and maintain strict citation accuracy.\n"
    "Cite evidence as [doc:page:start-end]. If no sufficient evidence exists, reply: 'No strong match found'.\n"
    "Prioritize sources with higher relevance scores. Be concise but comprehensive.\n"
    "When multiple sources support a claim, cite the most authoritative ones.\n"
)

def _compose_context(matches: List[dict]) -> str:
    lines = []
    for m in matches:
        md = m["metadata"]
        span = md.get("span", {})
        cite = f"{md.get('file','unknown')}:{span.get('page','?')}:{span.get('start','?')}-{span.get('end','?')}"
        lines.append(f"[{cite}] {md.get('text','')}\n-- parent: {md.get('parent_excerpt','')[:400]}")
    return "\n\n".join(lines)


def _compose_context_enhanced(matches: List[dict]) -> str:
    """
    Enhanced context composition for MoE pipeline with relevance scores and better formatting.
    """
    if not matches:
        return ""

    lines = []
    for i, match in enumerate(matches):
        metadata = match.get("metadata", {})
        span = metadata.get("span", {})

        # Enhanced citation format
        cite = f"{metadata.get('file','unknown')}:{span.get('page','?')}:{span.get('start','?')}-{span.get('end','?')}"

        # Include relevance scores if available
        scores_info = ""
        if 'cross_encoder_score' in match:
            scores_info = f" [rel:{match['cross_encoder_score']:.3f}]"
        elif 'score' in match:
            scores_info = f" [sim:{match['score']:.3f}]"

        text = metadata.get('text', '')
        parent = metadata.get('parent_excerpt', '')[:400]

        lines.append(
            f"[{cite}]{scores_info} {text}\n"
            f"-- context: {parent}"
        )

    return "\n\n".join(lines)

def _call_llm(cfg, system: str, question: str, context: str) -> str:
    """Secure LLM API call with comprehensive security controls"""
    url = "https://openrouter.ai/api/v1/chat/completions"

    # Use secure API headers
    headers = get_secure_api_headers("openrouter", cfg.OPENROUTER_API_KEY)
    headers.update({
        "HTTP-Referer": cfg.OPENROUTER_REFERER,
        "X-Title": cfg.OPENROUTER_TITLE,
    })

    payload = {
        "model": cfg.OPENROUTER_MODEL,
        "messages": [
            {"role":"system","content": system},
            {"role":"user","content": f"Question: {question}\\n\\nSources:\\n{context}"}
        ],
        "temperature": 0.0,
        "max_tokens": getattr(cfg, 'MAX_RESPONSE_TOKENS', 1000)
    }

    try:
        log_security_event("API_REQUEST_STARTED", {
            "service": "openrouter",
            "model": cfg.OPENROUTER_MODEL,
            "question_length": len(question),
            "context_length": len(context)
        }, "INFO")

        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()

        response = r.json()["choices"][0]["message"]["content"]

        log_security_event("API_REQUEST_COMPLETED", {
            "service": "openrouter",
            "response_length": len(response),
            "status_code": r.status_code
        }, "INFO")

        return response

    except requests.exceptions.RequestException as e:
        log_security_event("API_REQUEST_FAILED", {
            "service": "openrouter",
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
        }, "ERROR")
        raise
    except Exception as e:
        log_security_event("API_REQUEST_ERROR", {
            "service": "openrouter",
            "error": str(e)
        }, "ERROR")
        raise

def rag_chat(cfg, embedder, message: str, history: List[Tuple[str,str]]):
    """
    Enhanced RAG chat function with MoE integration support.

    Uses MoE pipeline when enabled, falls back to traditional RAG otherwise.
    """
    try:
        # Check if MoE is enabled
        moe_config = get_moe_config()
        if moe_config.enabled:
            return _rag_chat_with_moe(cfg, embedder, message, history)
        else:
            return _rag_chat_traditional(cfg, embedder, message, history)
    except Exception as e:
        # Fallback to traditional RAG on any MoE errors
        print(f"MoE pipeline failed, falling back to traditional RAG: {e}")
        return _rag_chat_traditional(cfg, embedder, message, history)


def _rag_chat_traditional(cfg, embedder, message: str, history: List[Tuple[str,str]]) -> str:
    """
    Traditional RAG implementation (original behavior).
    """
    qvec = embedder.encode(message, normalize_embeddings=True).tolist()
    res = vector_query(cfg, qvec, top_k=cfg.TOP_K, namespace=cfg.NAMESPACE)
    matches = [{"id":m["id"], "score":m["score"], "metadata": m.get("metadata",{})} for m in res.get("matches", [])]
    ctx = _compose_context(matches)
    if not matches:
        return "No strong match found."
    answer = _call_llm(cfg, SYSTEM_PROMPT, message, ctx)
    return answer


def _rag_chat_with_moe(cfg, embedder, message: str, history: List[Tuple[str,str]]) -> str:
    """
    Enhanced RAG with MoE pipeline integration.
    """
    try:
        # Get MoE pipeline
        moe_pipeline = get_moe_pipeline()

        # Prepare retrieval function for MoE pipeline
        def retrieval_function(query=None, query_embedding=None, top_k=10, namespace=None, **kwargs):
            """Wrapper for vectorstore query function"""
            if query_embedding is None and query is not None:
                query_embedding = embedder.encode([query], normalize_embeddings=True)[0].tolist()
            elif query_embedding is not None and hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            # Apply expert filtering if provided
            filter_dict = {}
            if "expert_filter" in kwargs and kwargs["expert_filter"]:
                filter_dict = {"expert_id": {"$in": kwargs["expert_filter"]}}

            # Ensure query_embedding is a list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Ensure we have a valid embedding
            if query_embedding is None:
                raise ValueError("No query embedding provided")

            return vector_query(
                cfg,
                query_embedding,
                top_k=top_k,
                namespace=namespace or cfg.NAMESPACE,
                filter=filter_dict
            )

        # Prepare generation function for MoE pipeline
        def generation_function(query=None, context=None, retrieval_matches=None):
            """Wrapper for LLM generation"""
            if not retrieval_matches:
                return "No strong match found."

            # Use enhanced context composition for MoE
            ctx = _compose_context_enhanced(retrieval_matches)
            # Ensure query is not None
            query_text = query if query is not None else message
            return _call_llm(cfg, ENHANCED_SYSTEM_PROMPT, query_text, ctx)

        # Process query through MoE pipeline
        result = moe_pipeline.process_query(
            query=message,
            retrieval_function=retrieval_function,
            generation_function=generation_function
        )

        # Return the final answer
        if result.final_answer:
            return result.final_answer
        elif result.retrieval_matches:
            # Fallback: generate answer from matches
            ctx = _compose_context_enhanced(result.retrieval_matches)
            return _call_llm(cfg, ENHANCED_SYSTEM_PROMPT, message, ctx)
        else:
            return "No strong match found."

    except Exception as e:
        print(f"MoE processing failed: {e}")
        # Fallback to traditional RAG
        return _rag_chat_traditional(cfg, embedder, message, history)
