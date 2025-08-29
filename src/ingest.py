import hashlib
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from .parsers import parse_any
from .propositionizer import propositionize_paragraphs
from .vectorstore import upsert_props

def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def ingest_files(cfg, embedder, files: List[Path], namespace: str) -> Dict:
    report = {"namespace": namespace, "files": []}
    for f in tqdm(files, desc="Ingest"):
        info = {"file": Path(f).name, "propositions": 0}
        parsed = parse_any(Path(f))
        doc_id = _hash(str(f))
        prop_batches = propositionize_paragraphs(cfg, parsed["paragraphs"])
        vectors = []
        prop_count = 0
        for batch in prop_batches:
            parent = batch.get("parent_excerpt","")
            for i, p in enumerate(batch.get("propositions", [])):
                text = p.get("text","").strip()
                if not text:
                    continue
                vec = embedder.encode(text, normalize_embeddings=True).tolist()
                pid = f"{doc_id}:{_hash(text)}"
                vectors.append({
                    "id": pid,
                    "values": vec,
                    "metadata": {
                        "doc_id": doc_id,
                        "file": Path(f).name,
                        "text": text,
                        "parent_excerpt": parent,
                        "span": p.get("span", {}),
                        "tags": p.get("tags", []),
                    }
                })
                prop_count += 1
        if vectors:
            upsert_props(cfg, vectors, namespace)
        info["propositions"] = prop_count
        report["files"].append(info)
    return report
