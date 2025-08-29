import math, json
from typing import List, Dict

def _idset(s: str) -> set:
    return set([x.strip() for x in (s or "").split(";") if x.strip()])

def hit_at_k(relevant_ids: str, predicted_ids: str) -> float:
    rel = _idset(relevant_ids)
    pred = [x for x in (predicted_ids or "").split(";") if x]
    return 1.0 if any(p in rel for p in pred) else 0.0

def ndcg_at_k(relevant_ids: str, predicted_ids: str) -> float:
    rel = _idset(relevant_ids)
    pred = [x for x in (predicted_ids or "").split(";") if x]
    dcg = sum((1.0 / math.log2(i+2)) for i,p in enumerate(pred) if p in rel)
    ideal = sum((1.0 / math.log2(i+2)) for i in range(min(len(rel), len(pred))))
    return dcg / ideal if ideal > 0 else 0.0

def span_accuracy(true_spans: List[Dict], pred_spans: List[Dict]) -> float:
    # Jaccard over character indices unioned across spans
    def to_set(spans):
        S = set()
        for sp in spans or []:
            a, b = int(sp.get("start",0)), int(sp.get("end",0))
            S.update(range(max(0,a), max(0,b)))
        return S
    T, P = to_set(true_spans), to_set(pred_spans)
    if not T or not P: return 0.0
    return len(T & P) / len(T | P)
