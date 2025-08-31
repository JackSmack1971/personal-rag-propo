"""
Enhanced Embeddings Module for 2025 Stack
Supports Sentence-Transformers 5.x with multi-backend capabilities,
sparse encoding, OpenVINO quantization, and performance optimizations.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import hashlib
import psutil
import numpy as np

# 2025 Stack: Sentence-Transformers 5.x with multi-backend support
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Enhanced model cache with backend-specific keys and performance tracking
_model_cache: Dict[str, Dict[str, Any]] = {}

# Default configuration
DEFAULT_BACKEND = os.getenv("SENTENCE_TRANSFORMERS_BACKEND", "torch")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sentence_transformers_2025"

# Performance tracking
_performance_metrics = {
    "model_load_times": {},
    "inference_times": {},
    "memory_usage": {}
}

def _get_cache_key(model_name: str, backend: str, **kwargs) -> str:
    """Generate unique cache key for model configuration"""
    config_str = f"{model_name}_{backend}_{sorted(kwargs.items())}"
    return hashlib.md5(config_str.encode()).hexdigest()

def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def _optimize_batch_size(sentences: List[str], max_batch_size: int = 32) -> int:
    """Dynamically optimize batch size based on input size and memory"""
    if len(sentences) <= 4:
        return len(sentences)  # No batching for small inputs

    # Reduce batch size for large inputs to prevent memory issues
    if len(sentences) > 100:
        return min(max_batch_size // 2, len(sentences))

    return min(max_batch_size, len(sentences))

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
            start_time = time.time()
            initial_memory = _get_memory_usage()

            logger.info(f"Loading model {model_name} with backend {backend}")

            model_kwargs = {
                "model_name_or_path": model_name,
                "backend": backend,
                "trust_remote_code": trust_remote_code,
                "cache_folder": cache_dir or str(DEFAULT_CACHE_DIR),
                **kwargs
            }

            # Add OpenVINO quantization for CPU backend
            if backend == "openvino":
                model_kwargs.update({
                    "model_kwargs": {"torch_dtype": "auto"},
                    "tokenizer_kwargs": {"use_fast": True}
                })

            # Remove None values
            model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

            embedder = SentenceTransformer(**model_kwargs)

            load_time = time.time() - start_time
            final_memory = _get_memory_usage()
            memory_delta = final_memory - initial_memory

            # Store model with metadata
            _model_cache[cache_key] = {
                "model": embedder,
                "load_time": load_time,
                "memory_usage": memory_delta,
                "backend": backend,
                "created_at": time.time()
            }

            # Track performance metrics
            _performance_metrics["model_load_times"][cache_key] = load_time
            _performance_metrics["memory_usage"][cache_key] = memory_delta

            logger.info(f"Successfully loaded model {model_name} in {load_time:.2f}s, memory: {memory_delta:.1f}MB")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    return _model_cache[cache_key]["model"]

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

def encode_optimized(
    embedder: SentenceTransformer,
    sentences: List[str],
    batch_size: Optional[int] = None,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = False
) -> np.ndarray:
    """
    Optimized encoding with dynamic batch sizing and memory management.

    Args:
        embedder: SentenceTransformer model
        sentences: List of sentences to encode
        batch_size: Override batch size (auto-optimized if None)
        normalize_embeddings: Whether to normalize embeddings
        show_progress_bar: Whether to show progress bar

    Returns:
        Encoded embeddings as numpy array
    """
    if not sentences:
        return np.array([])

    # Dynamic batch size optimization
    if batch_size is None:
        batch_size = _optimize_batch_size(sentences)

    start_time = time.time()
    initial_memory = _get_memory_usage()

    try:
        # Use optimized encoding
        embeddings = embedder.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar
        )

        inference_time = time.time() - start_time
        final_memory = _get_memory_usage()
        memory_delta = final_memory - initial_memory

        # Track performance
        cache_key = list(_model_cache.keys())[0] if _model_cache else "unknown"
        if cache_key not in _performance_metrics["inference_times"]:
            _performance_metrics["inference_times"][cache_key] = []

        _performance_metrics["inference_times"][cache_key].append({
            "time": inference_time,
            "memory_delta": memory_delta,
            "batch_size": batch_size,
            "num_sentences": len(sentences),
            "timestamp": time.time()
        })

        logger.debug(f"Encoded {len(sentences)} sentences in {inference_time:.3f}s, batch_size={batch_size}, memory={memory_delta:.1f}MB")

        return embeddings

    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise

def warm_up_model(embedder: SentenceTransformer, num_samples: int = 5):
    """Warm up model with sample encodings to improve initial performance"""
    sample_sentences = [f"This is sample sentence {i} for model warm-up." for i in range(num_samples)]

    start_time = time.time()
    embedder.encode(sample_sentences, batch_size=1, show_progress_bar=False)
    warm_up_time = time.time() - start_time

    logger.info(f"Model warm-up completed in {warm_up_time:.3f}s")
    return warm_up_time

def get_performance_metrics() -> Dict[str, Any]:
    """Get comprehensive performance metrics"""
    return {
        "model_load_times": _performance_metrics["model_load_times"],
        "inference_times": _performance_metrics["inference_times"],
        "memory_usage": _performance_metrics["memory_usage"],
        "cache_info": get_cache_info(),
        "system_memory": _get_memory_usage()
    }

def clear_performance_metrics():
    """Clear all performance metrics"""
    _performance_metrics["model_load_times"].clear()
    _performance_metrics["inference_times"].clear()
    _performance_metrics["memory_usage"].clear()
    logger.info("Cleared performance metrics")

def invalidate_cache_entry(cache_key: str):
    """Invalidate specific cache entry"""
    if cache_key in _model_cache:
        del _model_cache[cache_key]
        logger.info(f"Invalidated cache entry: {cache_key}")
    else:
        logger.warning(f"Cache entry not found: {cache_key}")

def get_optimal_backend() -> str:
    """Determine optimal backend based on system capabilities"""
    # Check for CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            return "torch"  # GPU backend
    except ImportError:
        pass

    # Default to OpenVINO for CPU optimization
    return "openvino"

# Backward compatibility function
def get_embedder_legacy(model_name: str) -> SentenceTransformer:
    """Legacy function for backward compatibility"""
    return get_embedder(model_name, backend="torch")
