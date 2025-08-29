import os, json, requests
from typing import List, Tuple
from .vectorstore import query

SYSTEM_PROMPT = (
    "You are a precise research assistant.\n"
    "Use ONLY the provided sources.\n"
    "Cite evidence as [doc:page:start-end]. If no sufficient evidence, reply: 'No strong match found'.\n"
    "Be concise.\n"
)

def _compose_context(matches: List[dict]) -> str:
    lines = []
    for m in matches:
        md = m["metadata"]
        span = md.get("span", {})
        cite = f"{md.get('file','unknown')}:{span.get('page','?')}:{span.get('start','?')}-{span.get('end','?')}"
        lines.append(f"[{cite}] {md.get('text','')}\n-- parent: {md.get('parent_excerpt','')[:400]}")
    return "\n\n".join(lines)

def _call_llm(cfg, system: str, question: str, context: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": cfg.OPENROUTER_REFERER,
        "X-Title": cfg.OPENROUTER_TITLE,
    }
    payload = {
        "model": cfg.OPENROUTER_MODEL,
        "messages": [
            {"role":"system","content": system},
            {"role":"user","content": f"Question: {question}\\n\\nSources:\\n{context}"}
        ],
        "temperature": 0.0
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def rag_chat(cfg, embedder, message: str, history: List[Tuple[str,str]]):
    qvec = embedder.encode(message, normalize_embeddings=True).tolist()
    res = query(cfg, qvec, top_k=cfg.TOP_K, namespace=cfg.NAMESPACE)
    matches = [{"id":m["id"], "score":m["score"], "metadata": m.get("metadata",{})} for m in res.get("matches", [])]
    ctx = _compose_context(matches)
    if not matches:
        return "No strong match found."
    answer = _call_llm(cfg, SYSTEM_PROMPT, message, ctx)
    return answer
