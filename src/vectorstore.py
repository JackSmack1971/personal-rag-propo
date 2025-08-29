from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec

def _client(cfg) -> Pinecone:
    return Pinecone(api_key=cfg.PINECONE_API_KEY)

def ensure_index(cfg, dim: int):
    pc = _client(cfg)
    exists = any(ix.name == cfg.PINECONE_INDEX for ix in pc.list_indexes())
    if not exists:
        pc.create_index(
            name=cfg.PINECONE_INDEX,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cfg.PINECONE_CLOUD, region=cfg.PINECONE_REGION),
        )

def upsert_props(cfg, vectors: List[Dict], namespace: str):
    pc = _client(cfg)
    index = pc.Index(cfg.PINECONE_INDEX)
    index.upsert(vectors=vectors, namespace=namespace)

def query(cfg, vector: list, top_k: int, namespace: str):
    pc = _client(cfg)
    index = pc.Index(cfg.PINECONE_INDEX)
    return index.query(vector=vector, top_k=top_k, include_metadata=True, namespace=namespace)
