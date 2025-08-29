import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str
    OPENROUTER_REFERER: str
    OPENROUTER_TITLE: str

    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_CLOUD: str
    PINECONE_REGION: str

    EMBED_MODEL: str
    NAMESPACE: str
    TOP_K: int

    @classmethod
    def from_env(cls):
        return cls(
            OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY",""),
            OPENROUTER_MODEL=os.getenv("OPENROUTER_MODEL","openrouter/auto"),
            OPENROUTER_REFERER=os.getenv("OPENROUTER_REFERER","http://localhost:7860"),
            OPENROUTER_TITLE=os.getenv("OPENROUTER_TITLE","Personal RAG (Propositional)"),
            PINECONE_API_KEY=os.getenv("PINECONE_API_KEY",""),
            PINECONE_INDEX=os.getenv("PINECONE_INDEX","personal-rag"),
            PINECONE_CLOUD=os.getenv("PINECONE_CLOUD","aws"),
            PINECONE_REGION=os.getenv("PINECONE_REGION","us-east-1"),
            EMBED_MODEL=os.getenv("EMBED_MODEL","BAAI/bge-small-en-v1.5"),
            NAMESPACE=os.getenv("NAMESPACE","default"),
            TOP_K=int(os.getenv("TOP_K","6")),
        )
