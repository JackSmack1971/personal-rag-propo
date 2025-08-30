import os
from dataclasses import dataclass, field
from typing import Optional

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

    # ---- 2025 Stack Configuration ----
    # Sentence-Transformers backend (torch/onnx/openvino)
    SENTENCE_TRANSFORMERS_BACKEND: str = "torch"

    # Gradio 5.x settings
    GRADIO_ANALYTICS_ENABLED: bool = False
    GRADIO_SSR_ENABLED: bool = True

    # MoE Configuration (2025 features)
    moe_enabled: bool = False
    moe_config: Optional[dict] = None

    # Security settings
    TRUST_REMOTE_CODE: bool = False
    MAX_FILE_SIZE_MB: int = 10

    # Performance settings
    ENABLE_SPARSE_ENCODING: bool = False
    CACHE_EMBEDDINGS: bool = True

    @classmethod
    def from_env(cls):
        # Load MoE configuration if available
        moe_enabled = os.getenv("MOE_ENABLED", "false").lower() == "true"
        moe_config = None

        if moe_enabled:
            try:
                from .moe.config import get_moe_config
                moe_config_obj = get_moe_config()
                moe_config = moe_config_obj.to_dict()
            except Exception as e:
                print(f"Warning: Could not load MoE configuration: {e}")
                moe_enabled = False

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

            # 2025 Stack Configuration
            SENTENCE_TRANSFORMERS_BACKEND=os.getenv("SENTENCE_TRANSFORMERS_BACKEND","torch"),
            GRADIO_ANALYTICS_ENABLED=os.getenv("GRADIO_ANALYTICS_ENABLED","false").lower() == "true",
            GRADIO_SSR_ENABLED=os.getenv("GRADIO_SSR_ENABLED","true").lower() == "true",

            # Security settings
            TRUST_REMOTE_CODE=os.getenv("TRUST_REMOTE_CODE","false").lower() == "true",
            MAX_FILE_SIZE_MB=int(os.getenv("MAX_FILE_SIZE_MB","10")),

            # Performance settings
            ENABLE_SPARSE_ENCODING=os.getenv("ENABLE_SPARSE_ENCODING","false").lower() == "true",
            CACHE_EMBEDDINGS=os.getenv("CACHE_EMBEDDINGS","true").lower() == "true",

            moe_enabled=moe_enabled,
            moe_config=moe_config,
        )
