"""
Enhanced Embeddings Module for 2025 Stack
Supports Sentence-Transformers 5.x with multi-backend capabilities,
sparse encoding, and performance optimizations.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import hashlib

# 2025 Stack: Sentence-Transformers 5.x with multi-backend support
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Enhanced model cache with backend-specific keys
_model_cache: Dict[str, SentenceTransformer] = {}

# Default configuration
DEFAULT_BACKEND = os.getenv("SENTENCE_TRANSFORMERS_BACKEND", "torch")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sentence_transformers_2025"

def _get_cache_key(model_name: str, backend: str, **kwargs) -> str:
    """Generate unique cache key for model configuration"""
    config_str = f"{model_name}_{backend}_{sorted(kwargs.items())}"
    return hashlib.md5(config_str.encode()).hexdigest()

def get_embedder(
    model_name: str,
    backend: str = DEFAULT_BACKEND,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
) -> SentenceTransformer:
    """
    Get enhanced sentence transformer embedder with multi-backend support.

    Args:
        model_name: HuggingFace model name
        backend: Backend to use ('torch', 'onnx', 'openvino')
        cache_dir: Custom cache directory
        trust_remote_code: Whether to trust remote code (security)
        **kwargs: Additional model arguments

    Returns:
        Configured SentenceTransformer instance
    """
    cache_key = _get_cache_key(model_name, backend, **kwargs)

    if cache_key not in _model_cache:
        try:
            logger.info(f"Loading model {model_name} with backend {backend}")

            model_kwargs = {
                "model_name_or_path": model_name,
                "backend": backend,
                "trust_remote_code": trust_remote_code,
                "cache_folder": cache_dir or str(DEFAULT_CACHE_DIR),
                **kwargs
            }

            # Remove None values
            model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

            embedder = SentenceTransformer(**model_kwargs)
            _model_cache[cache_key] = embedder

            logger.info(f"Successfully loaded model {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    return _model_cache[cache_key]

# Sparse encoding functionality will be added when SparseEncoder becomes available
def get_sparse_encoder(*args, **kwargs):
    """
    Placeholder for sparse encoder functionality.
    Will be implemented when Sentence-Transformers sparse encoding is stable.
    """
    raise NotImplementedError("Sparse encoding not yet available in this version")

def clear_cache():
    """Clear all cached models"""
    _model_cache.clear()
    logger.info("Cleared all model caches")

def get_cache_info() -> Dict[str, Any]:
    """Get information about cached models"""
    return {
        "dense_models": len(_model_cache),
        "cache_keys": list(_model_cache.keys()),
        "cache_dir": str(DEFAULT_CACHE_DIR)
    }

# Backward compatibility function
def get_embedder_legacy(model_name: str) -> SentenceTransformer:
    """Legacy function for backward compatibility"""
    return get_embedder(model_name, backend="torch")
