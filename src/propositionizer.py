import os, json, time, requests
from typing import List, Dict
from .security import get_secure_api_headers, log_security_event

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
    if cfg.OPENROUTER_REFERER:
        hdr["HTTP-Referer"] = cfg.OPENROUTER_REFERER
    if cfg.OPENROUTER_TITLE:
        hdr["X-Title"] = cfg.OPENROUTER_TITLE
    return hdr

def propositionize_paragraphs(cfg, paragraphs: List[str]) -> List[Dict]:
    """Secure proposition extraction with comprehensive security controls"""
    url = "https://openrouter.ai/api/v1/chat/completions"

    # Use secure API headers
    headers = get_secure_api_headers("openrouter", cfg.OPENROUTER_API_KEY)
    headers.update({
        "HTTP-Referer": cfg.OPENROUTER_REFERER,
        "X-Title": cfg.OPENROUTER_TITLE,
    })

    out = []

    log_security_event("PROPOSITION_EXTRACTION_STARTED", {
        "paragraph_count": len(paragraphs),
        "model": cfg.OPENROUTER_MODEL
    }, "INFO")

    for i, para in enumerate(paragraphs):
        try:
            payload = {
                "model": cfg.OPENROUTER_MODEL,
                "messages": [
                    {"role":"system","content": SYS_PROMPT},
                    {"role":"user","content": para}
                ],
                "temperature": 0.0,
                "max_tokens": getattr(cfg, 'MAX_RESPONSE_TOKENS', 1000)
            }

            log_security_event("PROPOSITION_API_REQUEST", {
                "paragraph_index": i,
                "paragraph_length": len(para)
            }, "DEBUG")

            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()

            content = r.json()["choices"][0]["message"]["content"]

            # Secure JSON parsing with bounds checking
            start = content.find("{")
            end = content.rfind("}")

            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(content[start:end+1])
                except json.JSONDecodeError as e:
                    log_security_event("PROPOSITION_JSON_PARSE_ERROR", {
                        "error": str(e),
                        "content_length": len(content)
                    }, "WARNING")
                    data = {"propositions":[], "parent_excerpt": para}
            else:
                log_security_event("PROPOSITION_JSON_BOUNDS_ERROR", {
                    "start": start,
                    "end": end,
                    "content_length": len(content)
                }, "WARNING")
                data = {"propositions":[], "parent_excerpt": para}

            if "parent_excerpt" not in data:
                data["parent_excerpt"] = para

            out.append(data)

            log_security_event("PROPOSITION_EXTRACTED", {
                "paragraph_index": i,
                "propositions_count": len(data.get("propositions", []))
            }, "DEBUG")

        except requests.exceptions.RequestException as e:
            log_security_event("PROPOSITION_API_ERROR", {
                "paragraph_index": i,
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }, "ERROR")
            # Return safe fallback
            out.append({"propositions":[], "parent_excerpt": para})

        except Exception as e:
            log_security_event("PROPOSITION_PROCESSING_ERROR", {
                "paragraph_index": i,
                "error": str(e)
            }, "ERROR")
            # Return safe fallback
            out.append({"propositions":[], "parent_excerpt": para})

        # Rate limiting delay
        time.sleep(0.2)

    log_security_event("PROPOSITION_EXTRACTION_COMPLETED", {
        "total_paragraphs": len(paragraphs),
        "successful_extractions": len([d for d in out if d.get("propositions")])
    }, "INFO")

    return out
