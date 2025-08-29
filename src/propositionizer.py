import os, json, time, requests
from typing import List, Dict

SYS_PROMPT = (
    "You extract ATOMIC FACTS (propositions) from paragraphs.\n"
    "Rules:\n"
    " - Only restate facts explicitly present; DO NOT introduce new facts.\n"
    " - Keep each proposition <= 30 words, self-contained, declarative.\n"
    " - Return strict JSON: {\"propositions\": [{\"text\":\"...\",\"span\":{\"start\":int,\"end\":int},\"tags\":[\"...\"]}],\"parent_excerpt\":\"...\"}\n"
    " - Include character span [start,end) indices relative to the provided paragraph text.\n"
)

def _openrouter_headers(cfg):
    hdr = {
        "Authorization": f"Bearer {cfg.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    # Optional attribution headers
    if cfg.OPENROUTER_REFERER:
        hdr["HTTP-Referer"] = cfg.OPENROUTER_REFERER
    if cfg.OPENROUTER_TITLE:
        hdr["X-Title"] = cfg.OPENROUTER_TITLE
    return hdr

def propositionize_paragraphs(cfg, paragraphs: List[str]) -> List[Dict]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    out = []
    for para in paragraphs:
        payload = {
            "model": cfg.OPENROUTER_MODEL,
            "messages": [
                {"role":"system","content": SYS_PROMPT},
                {"role":"user","content": para}
            ],
            "temperature": 0.0,
        }
        r = requests.post(url, headers=_openrouter_headers(cfg), json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        # Be tolerant to formatting by finding first/last JSON block
        start = content.find("{")
        end = content.rfind("}")
        data = json.loads(content[start:end+1]) if start!=-1 and end!=-1 else {"propositions":[], "parent_excerpt":para}
        if "parent_excerpt" not in data:
            data["parent_excerpt"] = para
        out.append(data)
        time.sleep(0.2)  # mild pacing
    return out
